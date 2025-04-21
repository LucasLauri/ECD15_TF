import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, List, Dict

class ModelPromoter:
    """Classe para gerenciar a promoção de modelos de ML seguindo boas práticas."""
    
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        """Inicializa o cliente MLflow."""
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.staging_threshold = 0.56  # Limite mínimo de F1-score para staging
        self.best_model: Optional[int] = None
        self.best_f1_score: float = 0.0

    def get_model_versions(self, model_name: str) -> List[Dict]:
        """Obtém todas as versões de um modelo registrado."""
        return self.client.search_model_versions(f"name='{model_name}'")

    def evaluate_model(self, version: Dict) -> float:
        """Avalia um modelo e retorna seu F1-score."""
        run = self.client.get_run(version.run_id)
        return run.data.metrics.get("f1_score", 0.0)

    def transition_model_stage(self, model_name: str, version: int, stage: str, reason: str = ""):
        """Transiciona um modelo para um estágio específico com log."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Modelo versão {version} movido para {stage}. {reason}")

    def process_versions(self, model_name: str):
        """Processa todas as versões do modelo aplicando regras de promoção."""
        versions = self.get_model_versions(model_name)
        
        for version in versions:
            try:
                f1_score = self.evaluate_model(version)
                
                if f1_score > self.staging_threshold:
                    self.transition_model_stage(
                        model_name,
                        version.version,
                        "Staging",
                        f"F1-score: {f1_score:.4f}"
                    )
                    
                    # Atualiza o melhor modelo candidato a produção
                    if f1_score > self.best_f1_score:
                        self.best_f1_score = f1_score
                        self.best_model = version.version
                else:
                    self.transition_model_stage(
                        model_name,
                        version.version,
                        "Archived",
                        f"F1-score {f1_score:.4f} abaixo do threshold"
                    )
                    
            except Exception as e:
                print(f"Erro ao processar versão {version.version}: {str(e)}")

    def promote_champion(self, model_name: str):
        """Promove o melhor modelo para produção."""
        if self.best_model:
            self.transition_model_stage(
                model_name,
                self.best_model,
                "Production",
                f"Champion com F1-score {self.best_f1_score:.4f}"
            )
        else:
            print("Nenhum modelo atende aos critérios para produção.")

def main():
    """Função principal."""
    promoter = ModelPromoter()
    model_name = "RandomForestGridSearch"
    
    try:
        promoter.process_versions(model_name)
        promoter.promote_champion(model_name)
    except Exception as e:
        print(f"Erro no pipeline de promoção: {str(e)}")

if __name__ == "__main__":
    main()