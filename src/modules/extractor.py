from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import re

class TableParser:
    def __init__(self, model_path, device="cuda"):
        print(f"[Extractor] Loading Donut from {model_path}...")
        self.processor = DonutProcessor.from_pretrained(model_path)
        # Half precision for 8GB GPU
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_path, 
            torch_dtype=torch.float16
        )
        self.model.to(device)
        self.device = device

    def extract_table(self, image_crop):
        """
        Input: PIL Image of just the table
        Output: JSON Dict
        """
        pixel_values = self.processor(image_crop, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device).half()
        
        # Prepare Prompt (Start Token)
        task_prompt = "<s>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # Generate
        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=768,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            repetition_penalty=1.2
        )
        
        # Decode
        seq = self.processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()
        
        return self._token2json(seq)

    def _token2json(self, tokens, is_inner_value=False):
        # (Standard Donut JSON parsing logic)
        output = {}
        while tokens:
            start_token = re.search(r"<([a-zA-Z0-9_]+)>", tokens)
            if start_token:
                key = start_token.group(1)
                end_token = re.search(f"</{key}>", tokens)
                if end_token:
                    start_index = start_token.end()
                    end_index = end_token.start()
                    value = tokens[start_index:end_index].strip()
                    if re.search(r"<.*?>", value):
                        value = self._token2json(value, is_inner_value=True)
                        if isinstance(value, list):
                            if key not in output: output[key] = []
                            output[key].extend(value)
                        else:
                            if key in output:
                                if not isinstance(output[key], list): output[key] = [output[key]]
                                output[key].append(value)
                            else:
                                output[key] = value
                    else:
                        output[key] = value
                    tokens = tokens[end_token.end():]
                else:
                    tokens = ""
            else:
                tokens = ""
        return output