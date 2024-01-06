PROJECT_NAME = ProbablyOverthinkingIt
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python


## Set up python interpreter environment
create_environment:
	conda create -y --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


tests:
	cd notebooks; pytest --nbmake gaussian.ipynb
	cd notebooks; pytest --nbmake inspection.ipynb
	cd notebooks; pytest --nbmake preston.ipynb
	cd notebooks; pytest --nbmake lognormal.ipynb
	cd notebooks; pytest --nbmake nbue.ipynb
	cd notebooks; pytest --nbmake berkson.ipynb
	cd notebooks; pytest --nbmake longtail.ipynb
	cd notebooks; pytest --nbmake overton.ipynb
