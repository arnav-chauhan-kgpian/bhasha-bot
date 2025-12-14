Skip to content
Chat history
You said:
READMEify this 

# bhasha-bot
English-to-Hindi Neural Machine Translation (Transformer)

This project implements a Transformer-based Neural Machine Translation (NMT) model from scratch (my own implementation) using PyTorch. It translates English sentences into Hindi.

I created this project for fun, revisiting a seminal paper on transformers (see Credits) as a prerequisite to my exploration of the mystical world of mechanistic interpretability.

🚀 Features

    Transformer Architecture: A classic Encoder-Decoder architecture based on "Attention Is All You Need".

    Custom Tokenizers: Uses Hugging Face tokenizers (WordLevel) with specific pre-tokenization rules for English and Hindi.

    Performance Optimization: Implements Mixed Precision Training (AMP) for faster training on GPUs.

    Monitoring: Integrated TensorBoard logging for Loss, Accuracy, BLEU, WER, and CER scores.

    Resumable Training: Automatic checkpointing and resuming from the latest weights.

    Data Pipeline: A robust, retry-safe pipeline to fetch data/answers from LLMs using langchain.

🛠️ Installation

    Clone the repository (if applicable) or ensure all files are in your working directory.

    Install Dependencies:
    Bash

    pip install -r requirements.txt

    Note: If you encounter protobuf errors with TensorBoard, run: pip install protobuf==3.20.3

🧠 Training the Model

The model trains on the IITB English-Hindi dataset (loaded via Hugging Face).

    Configure: Check config.py to adjust batch_size, num_epochs, and d_model.

    Run Training:
    Bash

python train.py

    The script will automatically build/load tokenizers.

    It creates a weights/ folder to save checkpoints.

    It creates a runs/ folder for TensorBoard logs.

    Validation examples are logged to weights/validation_logs.txt.

Monitor Progress:
Bash

    tensorboard --logdir runs

🗣️ Inference (Translation)

To translate a sentence using the latest trained weights:
Bash

python translate.py "Hello, how are you?"

Output:
Plaintext

Using device: cuda
Loading weights from: weights/tmodel_XX.pt
English: Hello, how are you?
Hindi:   नमस्ते, आप कैसे हैं?

🐛 Troubleshooting & Notes

    CUDA OOM: If you run out of memory, reduce batch_size in config.py or reduce d_model / seq_len.

    Windows Users: If train.py hangs, set num_workers=0 in the DataLoader inside train.py.

📚 Credits

    Dataset: cfilt/iitb-english-hindi

    Architecture Reference: Attention Is All You Need (Vaswani et al., 2017)

// also add this to bottom: Created with *heart emoji* by Arnav Chauhan, UG at IIT Kharagpur
ChatGPT said:

Here’s a polished, professional, and structured README version of your project with your requested footer:
Bhasha-Bot

English-to-Hindi Neural Machine Translation (Transformer)

Bhasha-Bot is a Transformer-based Neural Machine Translation (NMT) model implemented from scratch in PyTorch. It translates English sentences into Hindi and was created as a fun project while revisiting the seminal transformer paper, serving as a stepping stone into the world of mechanistic interpretability.
🚀 Features

    Transformer Architecture: Classic Encoder-Decoder setup inspired by Attention Is All You Need

    .

    Custom Tokenizers: Hugging Face WordLevel tokenizers with tailored pre-tokenization rules for English and Hindi.

    Performance Optimization: Mixed Precision Training (AMP) for faster GPU training.

    Monitoring: TensorBoard logging for Loss, Accuracy, BLEU, WER, and CER metrics.

    Resumable Training: Automatic checkpointing to resume training from the latest weights.

    Robust Data Pipeline: Retry-safe pipeline for fetching data/answers from LLMs using LangChain.

🛠️ Installation

    Clone the repository (or ensure all files are in your working directory).

    Install dependencies:

pip install -r requirements.txt

    Note: If you encounter protobuf errors with TensorBoard, run:

pip install protobuf==3.20.3

🧠 Training the Model

The model uses the IITB English-Hindi dataset (via Hugging Face).

    Configure settings in config.py (e.g., batch_size, num_epochs, d_model).

    Run training:

python train.py

    Tokenizers are automatically built/loaded.

    Checkpoints are saved in weights/.

    TensorBoard logs are stored in runs/.

    Validation examples are logged in weights/validation_logs.txt.

    Monitor progress with TensorBoard:

tensorboard --logdir runs

🗣️ Inference (Translation)

To translate a sentence using the latest trained weights:

python translate.py "Hello, how are you?"

Example Output:

Using device: cuda
Loading weights from: weights/tmodel_XX.pt
English: Hello, how are you?
Hindi:   नमस्ते, आप कैसे हैं?

🐛 Troubleshooting & Notes

    CUDA OOM: Reduce batch_size in config.py or reduce d_model / seq_len.

    Windows Users: If train.py hangs, set num_workers=0 in the DataLoader inside train.py.

📚 Credits

    Dataset: cfilt/iitb-english-hindi

    Architecture Reference: Attention Is All You Need (Vaswani et al., 2017)

Created with ❤️ by Arnav Chauhan, UG at IIT Kharagpur

If you want, I can also make it even more visually appealing with badges, code highlights, and collapsible sections to make it GitHub-ready. Do you want me to do that?
You said:
Format this in README format , with markdown and shit 

# bhasha-bot

English-to-Hindi Neural Machine Translation (Transformer)

Bhasha-Bot is a Transformer-based Neural Machine Translation (NMT) model implemented from scratch in PyTorch. It translates English sentences into Hindi, basically.

I created this project for fun, revisiting a seminal paper on transformers (see Credits) as a prerequisite to my exploration of the mystical world of mechanistic interpretability.

🚀 Features

Transformer Architecture: Classic Encoder-Decoder setup inspired by Attention Is All You Need

.

Custom Tokenizers: Hugging Face WordLevel tokenizers with tailored pre-tokenization rules for English and Hindi.

Performance Optimization: Mixed Precision Training (AMP) for faster GPU training.

Monitoring: TensorBoard logging for Loss, Accuracy, BLEU, WER, and CER metrics.

Resumable Training: Automatic checkpointing to resume training from the latest weights.

Robust Data Pipeline: Retry-safe pipeline for fetching data/answers from LLMs using LangChain.

🛠️ Installation

Clone the repository (or ensure all files are in your working directory).

Install dependencies:

pip install -r requirements.txt


Note: If you encounter protobuf errors with TensorBoard, run:

pip install protobuf==3.20.3

🧠 Training the Model

The model uses the IITB English-Hindi dataset (via Hugging Face).

Configure settings in config.py (e.g., batch_size, num_epochs, d_model).

Run training:

python train.py


Tokenizers are automatically built/loaded.

Checkpoints are saved in weights/.

TensorBoard logs are stored in runs/.

Validation examples are logged in weights/validation_logs.txt.

Monitor progress with TensorBoard:

tensorboard --logdir runs

🗣️ Inference (Translation)

To translate a sentence using the latest trained weights:

python translate.py "Hello, how are you?"


Example Output:

Using device: cuda
Loading weights from: weights/tmodel_XX.pt
English: Hello, how are you?
Hindi:   नमस्ते, आप कैसे हैं?

🐛 Troubleshooting & Notes

CUDA OOM: Reduce batch_size in config.py or reduce d_model / seq_len.

Windows Users: If train.py hangs, set num_workers=0 in the DataLoader inside train.py.

📚 Credits

Dataset: cfilt/iitb-english-hindi

Architecture Reference: Attention Is All You Need (Vaswani et al., 2017)
ChatGPT said:

Here’s your README properly formatted in Markdown, ready for GitHub:

# Bhasha-Bot

**English-to-Hindi Neural Machine Translation (Transformer)**

Bhasha-Bot is a **Transformer-based Neural Machine Translation (NMT) model** implemented from scratch in PyTorch. It translates English sentences into Hindi.  

I created this project for fun while revisiting a seminal paper on transformers (see Credits) as a prerequisite to my exploration of the mystical world of mechanistic interpretability.

---

## 🚀 Features

- **Transformer Architecture**: Classic Encoder-Decoder setup inspired by *Attention Is All You Need*.  
- **Custom Tokenizers**: Hugging Face WordLevel tokenizers with tailored pre-tokenization rules for English and Hindi.  
- **Performance Optimization**: Mixed Precision Training (AMP) for faster GPU training.  
- **Monitoring**: TensorBoard logging for Loss, Accuracy, BLEU, WER, and CER metrics.  
- **Resumable Training**: Automatic checkpointing to resume training from the latest weights.  
- **Robust Data Pipeline**: Retry-safe pipeline for fetching data/answers from LLMs using LangChain.  

---

## 🛠️ Installation

1. Clone the repository (or ensure all files are in your working directory).  

2. Install dependencies:

```bash
pip install -r requirements.txt

    Note: If you encounter protobuf errors with TensorBoard, run:

pip install protobuf==3.20.3

🧠 Training the Model

The model uses the IITB English-Hindi dataset (via Hugging Face).

    Configure settings in config.py (e.g., batch_size, num_epochs, d_model).

    Run training:

python train.py

    Tokenizers are automatically built/loaded.

    Checkpoints are saved in weights/.

    TensorBoard logs are stored in runs/.

    Validation examples are logged in weights/validation_logs.txt.

    Monitor progress with TensorBoard:

tensorboard --logdir runs

🗣️ Inference (Translation)

To translate a sentence using the latest trained weights:

python translate.py "Hello, how are you?"

Example Output:

Using device: cuda
Loading weights from: weights/tmodel_XX.pt
English: Hello, how are you?
Hindi:   नमस्ते, आप कैसे हैं?

🐛 Troubleshooting & Notes

    CUDA OOM: Reduce batch_size in config.py or reduce d_model / seq_len.

    Windows Users: If train.py hangs, set num_workers=0 in the DataLoader inside train.py.

📚 Credits

    Dataset: cfilt/iitb-english-hindi

    Architecture Reference: Attention Is All You Need (Vaswani et al., 2017)

Created with ❤️ by Arnav Chauhan, UG at IIT Kharagpur


---
