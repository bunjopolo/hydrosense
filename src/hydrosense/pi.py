"""Read-only PI Web API adapter. Secrets never enter persisted settings."""
from __future__ import annotations

from datetime import datetime
from urllib.parse import urlsplit
import math
import requests


class PIClient:
    def __init__(self, settings, secret=''):
        url = settings.get('url', '').rstrip('/')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use an HTTPS PI Web API base URL without credentials, query, or fragment.')
        self.url = url
        self.server = settings.get('data_server', '')
        self.mapping = settings.get('web_ids', {})
        if not isinstance(self.mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in self.mapping.items()):
            raise ValueError('Web ID overrides must map tag names to non-empty Web ID strings.')
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.verify = settings.get('ca_bundle') or True
        mode = settings.get('auth', 'basic')
        if mode == 'basic':
            self.session.auth = (settings.get('username', ''), secret)
        elif mode == 'bearer':
            self.session.headers['Authorization'] = 'Bearer ' + secret
        elif mode == 'windows':
            try:
                from requests_negotiate_sspi import HttpNegotiateAuth
                self.session.auth = HttpNegotiateAuth()
            except ImportError:
                raise ValueError('Windows authentication requires Windows and requests-negotiate-sspi. Use the service account running HydroSense.') from None
        elif mode != 'anonymous':
            raise ValueError('Unsupported authentication method.')
        self.ids = {}

    def get(self, path, params=None):
        try:
            response = self.session.get(self.url + path, params=params, timeout=(5, 20), allow_redirects=False)
            if response.status_code != 200:
                raise ValueError(f'PI Web API returned HTTP {response.status_code}. Check URL, access and authentication.')
            return response.json()
        except requests.RequestException:
            raise ValueError('PI request failed. Check network access, TLS trust and server availability.') from None

    def resolve(self, tags):
        for tag in tags:
            if tag in self.ids:
                continue
            if self.mapping.get(tag):
                self.ids[tag] = self.mapping[tag]
            else:
                if not self.server:
                    raise ValueError('Enter the PI Data Archive server or a Web ID for every tag.')
                point = self.get('/points', {'path': '\\\\' + self.server + '\\' + tag})
                if not point.get('WebId'):
                    raise ValueError(f'PI did not resolve tag: {tag}')
                self.ids[tag] = point['WebId']

    def snapshot(self, tags):
        self.resolve(tags)
        values, times, quality = {}, {}, {}
        # Bounded batches avoid long URLs on large tag lists.
        for start in range(0, len(tags), 40):
            selected = tags[start:start + 40]
            payload = self.get('/streamsets/value', [('webId', self.ids[t]) for t in selected])
            indexed = {item.get('WebId'): item.get('Value', {}) for item in payload.get('Items', [])}
            for tag in selected:
                item = indexed.get(self.ids[tag], {})
                value = item.get('Value')
                good = item.get('Good') is True and not item.get('Questionable', False) and not item.get('Substituted', False)
                numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                quality[tag] = good and numeric
                values[tag] = float(value) if quality[tag] else None
                try:
                    timestamp = datetime.fromisoformat(item['Timestamp'].replace('Z', '+00:00'))
                    if timestamp.tzinfo is not None:
                        times[tag] = timestamp
                except (KeyError, TypeError, ValueError):
                    quality[tag] = False
        return values, times, quality
