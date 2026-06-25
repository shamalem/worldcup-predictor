.PHONY: setup data train seed api web test docker

setup:        ## install backend + frontend deps
	cd backend && pip install -r requirements-dev.txt
	cd frontend && npm install

data:         ## download results.csv
	cd backend && python -m scripts.download_data

train:        ## train + compare models
	cd backend && python -m ml.train

seed:         ## seed the database
	cd backend && python -m scripts.seed_db

api:          ## run the FastAPI backend
	cd backend && uvicorn app.main:app --reload

web:          ## run the React dev server
	cd frontend && npm run dev

test:         ## run backend tests
	cd backend && pytest -q

docker:       ## build + run full stack
	docker compose up --build
