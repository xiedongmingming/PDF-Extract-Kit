#### install git lfs
before you begin, make sure git large file storage (git lfs) is installed on your system. install it using the following command:

```bash
git lfs install
```

#### download the model from hugging face
to download the `pdf-extract-Kit` model from hugging face, use the following command:

```bash
git lfs clone https://huggingface.co/wanderkid/PDF-Extract-Kit
```

ensure that git lfs is enabled during the clone to properly download all large files.



put [model files]() here:

```
./
├── Layout
│   ├── config.json
│   └── weights.pth
├── MFD
│   └── weights.pt
├── MFR
│   └── UniMERNet
│       ├── config.json
│       ├── preprocessor_config.json
│       ├── pytorch_model.bin
│       ├── README.md
│       ├── tokenizer_config.json
│       └── tokenizer.json
└── README.md
```