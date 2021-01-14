.PHONY: 0 1 2 3 4 5

all: 0 1 2 3 4 5

0: 
	python -m rastervision.pipeline.cli run_command /opt/src/code/output/pipeline-config.json analyze

1: 0
	python -m rastervision.pipeline.cli run_command /opt/src/code/output/pipeline-config.json chip

2: 1
	python -m rastervision.pipeline.cli run_command /opt/src/code/output/pipeline-config.json train

3: 2
	python -m rastervision.pipeline.cli run_command /opt/src/code/output/pipeline-config.json predict

4: 3
	python -m rastervision.pipeline.cli run_command /opt/src/code/output/pipeline-config.json eval

5: 4
	python -m rastervision.pipeline.cli run_command /opt/src/code/output/pipeline-config.json bundle

