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
