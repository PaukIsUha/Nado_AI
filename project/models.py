from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
from sklearn.preprocessing import LabelEncoder
from docx import Document


class Prediction:
    difference: str
    description: str
    compl: str


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        diff_model_path = './trained_model_diff2'

        self.diff_tokenizer = AutoTokenizer.from_pretrained(diff_model_path)
        self.diff_model = AutoModelForSeq2SeqLM.from_pretrained(diff_model_path).to(self.device)

        desc_model_path = './trained_model_diff2'

        self.desc_tokenizer = AutoTokenizer.from_pretrained(desc_model_path)
        self.desc_model = AutoModelForSeq2SeqLM.from_pretrained(desc_model_path).to(self.device)

        compl_model_path = './trained_classifier_compl2'

        self.coml_tokenizer = LongformerTokenizer.from_pretrained(compl_model_path)
        self.compl_model = LongformerForSequenceClassification.from_pretrained(compl_model_path).to(self.device)

        labels = ['FC', 'LC', 'NC', 'PC']

        self.le = LabelEncoder()
        self.le.fit(labels)

    def __inference_difference(self, ssts_text: str, hmi_text: str) -> str:
        input_text = f"SSTS: {ssts_text} </s> HMI: {hmi_text}"
        inputs = self.diff_tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=1024).to(self.device)

        outputs = self.diff_model.generate(
            inputs,
            max_length=150,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        description = self.diff_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return description

    def __inference_description(self, ssts_text: str, hmi_text: str) -> str:
        input_text = f"SSTS: {ssts_text} </s> HMI: {hmi_text}"
        inputs = self.desc_tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=1024).to(self.device)

        outputs = self.desc_model.generate(
            inputs,
            max_length=150,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        description = self.desc_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return description

    def __inference_level(self, ssts_text: str, hmi_text: str) -> str:
        input_text = f"SSTS: {ssts_text} [SEP] HMI: {hmi_text}"
        inputs = self.coml_tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=1024).to(self.device)
        with torch.no_grad():
            outputs = self.compl_model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

            predicted_label = self.le.inverse_transform([predicted_class_id])[0]
            return predicted_label

    def __inference(self, ssts_text: str, hmi_text: str) -> Prediction:
        pred = Prediction()
        pred.difference = self.__inference_difference(ssts_text, hmi_text)
        pred.description = self.__inference_description(ssts_text, hmi_text)
        pred.compl = self.__inference_level(ssts_text, hmi_text)

    @staticmethod
    def __read_docx(filename):
        try:
            doc = Document(filename)
            text = "\n".join([para.text for para in doc.paragraphs])
            text = text.split("\n")
            text = text[1:]
            return "\n".join(text)
        except:
            return "NAN_CODE"

    def fit(self, file_stss: str, file_hmi: str) -> Prediction:
        ssts_info = self.__read_docx(file_stss)
        hmi_info = self.__read_docx(file_hmi)

        return self.__inference(ssts_info, hmi_info)


