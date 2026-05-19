import os
import urllib.request
import zipfile
from tqdm import tqdm

def download_fasttext_embeddings(output_dir):
    """
    Baixar embeddings FastText pré-treinados em português
    """
    print("\n" + "="*70)
    print("BAIXANDO EMBEDDINGS FASTTEXT - PORTUGUÊS")
    print("="*70)
    
    # Criar diretório
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # URL do FastText português (cc.pt.300.vec)
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.vec.gz"
    gz_path = os.path.join(models_dir, "cc.pt.300.vec.gz")
    vec_path = os.path.join(models_dir, "cc.pt.300.vec")
    
    print(f"Baixando FastText embeddings de:")
    print(f"URL: {url}")
    print(f"Destino: {gz_path}")
    
    # Baixar arquivo
    try:
        # Mostrar progresso
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="FastText PT") as t:
            urllib.request.urlretrieve(url, gz_path, reporthook=t.update_to)
        
        print(f"✓ Download concluído: {gz_path}")
        
        # Descomprimir
        print("Descomprimindo arquivo...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(vec_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"✓ Arquivo descomprimido: {vec_path}")
        
        # Verificar arquivo
        if os.path.exists(vec_path):
            size_mb = os.path.getsize(vec_path) / (1024 * 1024)
            print(f"✓ Embeddings FastText prontos: {size_mb:.1f} MB")
            return vec_path
        else:
            print("❌ Erro na descompressão")
            return None
            
    except Exception as e:
        print(f"❌ Erro no download: {e}")
        return None

if __name__ == "__main__":
    import gzip
    import shutil
    
    output_dir = 'vectorization/word2vec'
    vec_path = download_fasttext_embeddings(output_dir)
    
    if vec_path:
        print(f"\n✓ Embeddings FastText prontos para uso!")
        print(f"Caminho: {vec_path}")
    else:
        print(f"\n❌ Falha no download dos embeddings")
