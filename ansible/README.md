# Ansible Deployment

Deploys the BCT Review Recommender Agent to an Ubuntu server using `uv` + systemd + Nginx + Let's Encrypt.

**Target:** `review-recommender-agent.tosinamuda.com`
**Runs as:** `www-data` (nginx user)
**App directory:** `/opt/review-recommender-agent`

---

## Prerequisites (on the server, once)

```bash
sudo apt install -y ansible
```

DNS A record for `review-recommender-agent.tosinamuda.com` must point to the server IP before SSL is issued.

---

## First-time install

```bash
# 1. Clone the repo to get the ansible files
git clone https://github.com/tosinamuda/review-recommender-agent.git ~/bct-deploy

# 2. Run the playbook (as a user with sudo)
sudo bash ~/bct-deploy/ansible/deploy.sh
# This will:
#   - install uv to /usr/local/bin
#   - clone the repo to /opt/review-recommender-agent (as www-data)
#   - run uv sync
#   - copy .env.example → .env
#   - install + start the systemd service
#   - configure nginx
#   - obtain SSL certificate via Certbot

# 3. Fill in secrets
sudo nano /opt/review-recommender-agent/.env
# Required:
#   OPENROUTER_API_KEY=sk-or-...
#   LM_MODEL=openrouter/openai/gpt-oss-120b

# 4. Restart to pick up the env
sudo systemctl restart review-recommender-agent

# 5. Verify
sudo systemctl status review-recommender-agent
sudo journalctl -u review-recommender-agent -f
```

---

## Subsequent re-runs (updates / redeployment)

The playbook is idempotent — safe to re-run at any time.

```bash
# Pull the latest playbook first
cd ~/bct-deploy && git pull

# Re-run
sudo bash ~/bct-deploy/ansible/deploy.sh
```

What happens on re-run:

| Task | Behaviour |
|---|---|
| Install system packages | `ok` — already installed |
| Install uv | **skipped** — `creates:` guard |
| Clone repo | `ok` — already cloned |
| Pull latest changes | **runs** — fetches latest code |
| `uv sync` | **runs** — installs any new deps |
| Create `.env` | **skipped** — `force: no` protects your secrets |
| Systemd unit | `ok` or `changed` if template changed |
| Certbot SSL | **skipped** — cert already exists |

After re-run the service restarts automatically if the unit file or dependencies changed.

---

## Pulling code updates without re-running the full playbook

```bash
cd /opt/review-recommender-agent
sudo -u www-data git fetch origin && sudo -u www-data git reset --hard origin/main
sudo systemctl restart review-recommender-agent
```

---

## Useful commands on the server

```bash
# Service status
sudo systemctl status review-recommender-agent

# Live logs
sudo journalctl -u review-recommender-agent -f

# Restart
sudo systemctl restart review-recommender-agent

# Edit secrets
sudo nano /opt/review-recommender-agent/.env

# Check nginx config
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

---

## After first successful start — optional HF cache lock

Once the embedding model (`BAAI/bge-small-en-v1.5`) has been downloaded and cached, add this to `.env` to prevent any further HuggingFace network calls:

```bash
sudo nano /opt/review-recommender-agent/.env
# Add: HF_HUB_OFFLINE=1
sudo systemctl restart review-recommender-agent
```
