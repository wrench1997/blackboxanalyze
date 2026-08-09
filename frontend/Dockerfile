# PG-388 display frontend. Supply a reviewed immutable Node base digest.
ARG NODE_BASE_IMAGE
FROM ${NODE_BASE_IMAGE} AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

FROM ${NODE_BASE_IMAGE} AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG PG388_API_URL=http://pg388-backend:8088
ENV PG388_API_URL=${PG388_API_URL}
RUN npm run build

FROM ${NODE_BASE_IMAGE} AS runner
WORKDIR /app
ENV NODE_ENV=production HOSTNAME=0.0.0.0 PORT=3000
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
