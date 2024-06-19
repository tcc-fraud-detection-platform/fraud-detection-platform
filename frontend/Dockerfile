# Etapa 1: Construir a aplicação Angular
FROM node:18 AS build

WORKDIR /app

# Copiar os arquivos de configuração do npm e instalar as dependências
COPY package*.json ./
RUN npm install

# Copiar todos os arquivos do projeto e construir a aplicação
COPY . .
RUN npm run build --prod

# Etapa 2: Servir a aplicação com Nginx
FROM nginx:alpine

# Copiar os arquivos construídos do estágio de build para o nginx
COPY --from=build /app/dist/fronttcc /usr/share/nginx/html

# Expor a porta 80 para acessar a aplicação
EXPOSE 80

# Comando para iniciar o nginx
CMD ["nginx", "-g", "daemon off;"]
