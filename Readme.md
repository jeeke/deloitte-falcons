#Build Guide

git add .
git commit -m commit-message
docker build -t app .
docker run -p 5000:5000 app