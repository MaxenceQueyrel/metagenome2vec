NS = 'maxence27'

VERSION_M2V ?= '2.0'
REPO_M2V = 'metagenome2vec'
NAME_M2V = 'metagenome2vec'

VERSION_CAMISIM ?= '2.0'
REPO_CAMISIM = 'camisim'
NAME_CAMISIM = 'camisim'

.PHONY: build_m2v push_m2v build_camisim push_cammisim

build_m2v:
	docker build -t $(NS)/$(REPO_M2V):$(VERSION_M2V) -f ./Docker/metagenome2vec/Dockerfile .

push_m2v:
	docker push $(NS)/$(REPO_M2V):$(VERSION_M2V)

build_camisim:
	docker build -t $(NS)/$(REPO_CAMISIM):$(VERSION_CAMISIM) -f ./Docker/CAMISIM/Dockerfile .

push_cammisim:
	docker push $(NS)/$(REPO_CAMISIM):$(VERSION_CAMISIM)


default: build_m2v
