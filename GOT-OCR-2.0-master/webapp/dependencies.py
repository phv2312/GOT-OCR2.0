from functools import lru_cache
import os
import time
from io import BytesIO
from abc import ABC, abstractmethod

from loguru import logger

import torch
from transformers import AutoTokenizer
from PIL import Image
from GOT.utils.utils import disable_torch_init
from GOT.model import GOTQwenForCausalLM
from GOT.utils.utils import KeywordsStoppingCriteria
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.conversation import conv_templates, SeparatorStyle


class BaseModel(ABC):
    @abstractmethod
    def process(self, image_in: str | bytes):
        ...
    
    @classmethod
    def get_pipeline(cls) -> "BaseModel":
        ...


class OCRModel(BaseModel):
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
    DEFAULT_IM_START_TOKEN = '<img>'
    DEFAULT_IM_END_TOKEN = '</img>'

    def __init__(self, model_name: str):
        disable_torch_init()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Run with device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = GOTQwenForCausalLM.from_pretrained(
            model_name, 
            low_cpu_mem_usage=True, 
            device_map="cpu", 
            use_safetensors=True, 
            pad_token_id=151643
        ).eval()
        self.model.to(device=self.device)
        
        _image_size = 1024 # Hard-code here, should not be changed, cause RunTimeError
        self.image_processor = BlipImageEvalProcessor(image_size=_image_size)
        self.image_processor_high = BlipImageEvalProcessor(image_size=_image_size)

    @classmethod
    @lru_cache(1)
    def get_pipeline(cls) -> "OCRModel":
        current_dir: str = os.path.dirname(__file__)

        # Automatically load weight dir
        weight_path: str = os.path.join(
            current_dir, 
            "../weights/GOT_weights"
        )
        assert os.path.isdir(weight_path)

        return cls(model_name=weight_path) 

    def load_image(self, image_file: bytes | str):
        if type(image_file) is str:
            image = Image.open(image_file).convert("RGB")
        elif type(image_file) is bytes:
            image = Image.open(BytesIO(image_file)).convert('RGB')
        else:
            raise NotImplementedError(f"Unknown input: {image_file}")

        return image

    def process(self, image_in: str | bytes, ocr_type: str = "ocr") -> str:
        """
        Read image from path or bytes
        Process, get prompt and run model
        """
        time_start = time.time()
        image: Image.Image = self.load_image(image_in)
        image_tensor, image_tensor_high = self.process_image(image)
        prompt = self.build_prompt(_type=ocr_type, bbox=None, color=None)

        result = self.forward(image_tensor, image_tensor_high, prompt)

        # Preview results
        preview_str = f"{result[:97]}..." if len(result) > 97 else result
        logger.debug(f"Got response: {preview_str}")

        time_elapsed = time.time() - time_start
        logger.info(f"Elapsed time: {'%.3f' % time_elapsed}")
        
        return result

    def process_image(self, image: Image.Image):
        """
        Image processing
        """
        image_tensor = self.image_processor(image).unsqueeze(0).to(self.device)
        image_tensor_high = self.image_processor_high(image.copy()).unsqueeze(0).to(self.device)
        return image_tensor, image_tensor_high

    def build_prompt(
        self, 
        _type: str, 
        image_token_len: str = 256, 
        bbox: str="", 
        color: str = ""
    ):
        """
        Create initial prompt based on ocr_type
        """
        if _type == "format":
            qs = "OCR with format: "
        else:
            qs = "OCR: "

        if bbox:
            qs = str(bbox) + ' ' + qs

        if color:
            qs = f"[{color}] {qs}"

        # Use image start and end tokens
        qs = (
            self.DEFAULT_IM_START_TOKEN + 
            self.DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + 
            self.DEFAULT_IM_END_TOKEN + 
            '\n' + qs
        )
        return qs

    def forward(
        self, 
        image_tensor: torch.Tensor, 
        image_tensor_high: torch.Tensor, 
        prompt: str, 
        conv_mode="mpt"
    ):
        """
        Detailing a converstation prompt and decode with Deep Model
        """

        # Detailing converation prompt
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)

        inputs = self.tokenizer([conv.get_prompt()])
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # Decoding steps
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=[(image_tensor, image_tensor_high)],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                max_new_tokens=4096,
                stopping_criteria=[stopping_criteria]
            )

        result = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        result = "\n".join(result.split(conv.sep))

        return result


if __name__ == "__main__":        
    image_path = "test.jpg"

    ocr_model = OCRModel.get_pipeline()
    print(ocr_model)

    result = ocr_model.process(
        image_path, ocr_type="ocr"
    )
    print(result)
