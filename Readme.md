#Local Build Guide

git add .
git commit -m commit-message
docker build -t app .
docker run -p 5000:5000 app

#Heroku Build Guide
git add .
git commit -m commit-message
heroku container:push web
heroku container:release web

#Heroku online build
heroku stack:set container
git push heroku master