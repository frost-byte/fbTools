# Brave DevTools MCP — Two-Device Setup

Connects a Claude Code session running on one machine to a Brave browser session running on another,
using the `chrome-devtools-mcp-for-brave` MCP server over a local network (or VPN).

## Typical setup

| Role | Machine |
|------|---------|
| **Browser host** | The device running Brave with remote debugging enabled |
| **Claude Code host** | The device running Claude Code (and typically ComfyUI) |

These can be the same machine (use `localhost`), but the common case is a laptop running the browser
while a desktop or server runs Claude Code.

## Brave configuration (browser host)

Launch Brave with the remote debugging port open:

```
brave --remote-debugging-port=9222
```

Or set it persistently in Brave's shortcut / launch config. The port must be reachable from the
Claude Code host — on a local network this means being on the same subnet with no firewall blocking
9222. For cross-network access (cloud droplet, VPN), route the traffic through a VPN (WireGuard,
Tailscale, OpenVPN) and use the VPN-assigned IP of the browser host instead of its LAN IP.

## Claude Code configuration (Claude Code host)

Add this block to `~/.claude.json` under `"mcpServers"`:

```json
"mcpServers": {
  "brave-devtools": {
    "type": "stdio",
    "command": "npx",
    "args": [
      "chrome-devtools-mcp-for-brave@latest",
      "--browserUrl=http://<BROWSER_DEVICE_IP>:9222"
    ]
  }
}
```

Replace `<BROWSER_DEVICE_IP>` with:
- The **LAN IP** of the browser host (e.g. `192.168.1.x`) for same-network setups
- The **VPN IP** (e.g. Tailscale `100.x.x.x` or OpenVPN subnet IP) for cross-network setups

### Node.js / npx

`npx` must be on the PATH of the Claude Code host. If Node is managed via `nvm`, you may need to
supply an explicit path:

```json
"env": {
  "PATH": "/home/<user>/.local/share/nvm/<version>/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
}
```

Omit the `env` block if `npx` is already on the system PATH.

## Local config reference

See `docs/brave_mcp.local.md` (gitignored) for the actual IPs and paths configured on this
machine. Copy that file as a template when moving to a new host.
