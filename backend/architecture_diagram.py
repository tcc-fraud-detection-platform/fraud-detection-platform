from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import Users
from diagrams.programming.framework import Angular, Flask
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.container import Docker

with Diagram("", show=True):
    users = Users("Usuários")

    with Cluster("Cluster Kubernetes"):
        with Cluster("Aplicações Fixas"):
            frontend = Angular("Frontend")
            backend = Flask("\nBackend")

        with Cluster("Pods Sob Demanda"):
            modules = [Docker("Módulo de Imagens"), Docker("Módulo de Áudio"), Docker("Módulo de Fake News")]
        
        # Organizamos o fluxo de conexão
        frontend >> Edge(color="blue") >> backend
        backend >> Edge(color="red", style="dashed") >> modules

    with Cluster("AWS RDS"):
        postgres = PostgreSQL("PostgreSQL")

    backend >> Edge(color="black") >> postgres
    modules >> Edge(color="green") >> postgres

    # Conexão de usuários
    users >> Edge(color="blue") >> frontend
    users << Edge(color="blue") << frontend
