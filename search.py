from litepali import LitePali

class PDFSearch:
    def __init__(self, index_dir, device="cpu"):
        self.retriever = LitePali(device=device)
        self.retriever.load_index(index_dir)
        self.retriever._load_model_and_processor()
    
    def search(self, query, k=3):
        return self.retriever.search(query, k=k)
    
    @property
    def image_embeddings(self):
        return self.retriever.image_embeddings