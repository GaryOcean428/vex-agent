/**
 * Vex Tool — GitHub
 *
 * Interact with GitHub repositories via the REST API.
 * Supports: list repos, get file, create issue, search code.
 */

import { config } from '../config';
import { VexTool, ToolResult } from './registry';

async function githubApi(
  path: string,
  method = 'GET',
  body?: unknown,
): Promise<{ ok: boolean; status: number; data: unknown }> {
  const resp = await fetch(`https://api.github.com${path}`, {
    method,
    headers: {
      Authorization: config.githubToken ? `Bearer ${config.githubToken}` : '',
      Accept: 'application/vnd.github+json',
      'User-Agent': 'VexAgent/1.0',
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await resp.json().catch(() => null);
  return { ok: resp.ok, status: resp.status, data };
}

export const githubTool: VexTool = {
  name: 'github',
  description:
    'Interact with GitHub. Actions: list_repos, get_file, create_issue, search_code.',
  parameters: {
    type: 'object',
    properties: {
      action: {
        type: 'string',
        enum: ['list_repos', 'get_file', 'create_issue', 'search_code'],
        description: 'The GitHub action to perform',
      },
      owner: { type: 'string', description: 'Repository owner' },
      repo: { type: 'string', description: 'Repository name' },
      path: { type: 'string', description: 'File path (for get_file)' },
      title: { type: 'string', description: 'Issue title (for create_issue)' },
      body: { type: 'string', description: 'Issue body (for create_issue)' },
      query: { type: 'string', description: 'Search query (for search_code)' },
    },
    required: ['action'],
  },

  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const action = args.action as string;

    try {
      switch (action) {
        case 'list_repos': {
          const owner = (args.owner as string) || 'GaryOcean428';
          const res = await githubApi(`/users/${owner}/repos?per_page=10&sort=updated`);
          if (!res.ok) return { success: false, output: '', error: `GitHub API ${res.status}` };
          const repos = (res.data as Array<{ full_name: string; description: string }>).map(
            (r) => `${r.full_name}: ${r.description || '(no description)'}`,
          );
          return { success: true, output: repos.join('\n') };
        }

        case 'get_file': {
          const owner = args.owner as string;
          const repo = args.repo as string;
          const filePath = args.path as string;
          const res = await githubApi(`/repos/${owner}/${repo}/contents/${filePath}`);
          if (!res.ok) return { success: false, output: '', error: `GitHub API ${res.status}` };
          const content = Buffer.from(
            (res.data as { content: string }).content,
            'base64',
          ).toString('utf-8');
          return { success: true, output: content.slice(0, 8000) };
        }

        case 'create_issue': {
          const owner = args.owner as string;
          const repo = args.repo as string;
          const res = await githubApi(`/repos/${owner}/${repo}/issues`, 'POST', {
            title: args.title,
            body: args.body,
          });
          if (!res.ok) return { success: false, output: '', error: `GitHub API ${res.status}` };
          return {
            success: true,
            output: `Issue created: ${(res.data as { html_url: string }).html_url}`,
          };
        }

        case 'search_code': {
          const query = args.query as string;
          const res = await githubApi(
            `/search/code?q=${encodeURIComponent(query)}&per_page=5`,
          );
          if (!res.ok) return { success: false, output: '', error: `GitHub API ${res.status}` };
          const items = (
            res.data as { items: Array<{ repository: { full_name: string }; path: string }> }
          ).items.map((i) => `${i.repository.full_name}/${i.path}`);
          return { success: true, output: items.join('\n') || 'No results' };
        }

        default:
          return { success: false, output: '', error: `Unknown action: ${action}` };
      }
    } catch (err) {
      return { success: false, output: '', error: (err as Error).message };
    }
  },
};
