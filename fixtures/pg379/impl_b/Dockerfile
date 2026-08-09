# Build with a pre-pulled, digest-pinned base image in the authorized lab.
# No npm install is needed: the runtime uses only Node's standard library.
ARG NODE_BASE_IMAGE=node:20.11.1-alpine3.19
FROM ${NODE_BASE_IMAGE}

LABEL org.opencontainers.image.title="PG-379 implementation B"
LABEL org.opencontainers.image.description="Loopback-only dynamic evaluator fixture"

WORKDIR /opt/pg379-impl-b
COPY package.json server.js manifest.json README.md ./

USER node
EXPOSE 8799
ENTRYPOINT ["node", "server.js"]
