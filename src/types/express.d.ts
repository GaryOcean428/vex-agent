/**
 * Minimal ambient declarations for Express v4.
 *
 * @types/express can't be resolved in this build environment.
 * This covers the subset of the Express API used by the proxy server.
 */

declare module "express" {
  import { IncomingMessage, Server, ServerResponse } from "http";

  export interface Request extends IncomingMessage {
    body: unknown;
    params: Record<string, string>;
    query: Record<string, string | string[] | undefined>;
    path: string;
    headers: IncomingMessage["headers"];
    setTimeout(ms: number): this;
  }

  export interface Response extends ServerResponse {
    json(body: unknown): this;
    status(code: number): this;
    send(body: string | Buffer): this;
    sendFile(path: string): void;
    redirect(url: string): void;
    setHeader(name: string, value: string | number | readonly string[]): this;
    write(
      chunk: string | Buffer,
      cb?: (error: Error | null | undefined) => void,
    ): boolean;
    end(): this;
    setTimeout(ms: number): this;
  }

  export type NextFunction = (err?: unknown) => void;

  export type RequestHandler = (
    req: Request,
    res: Response,
    next: NextFunction,
  ) => void;

  export interface IRouter {
    get(path: string, ...handlers: RequestHandler[]): this;
    post(path: string, ...handlers: RequestHandler[]): this;
    delete(path: string, ...handlers: RequestHandler[]): this;
    use(...handlers: Array<RequestHandler | IRouter>): this;
    use(path: string, ...handlers: Array<RequestHandler | IRouter>): this;
  }

  export interface Application extends IRouter {
    listen(port: number, hostname: string, callback?: () => void): Server;
    listen(port: number, callback?: () => void): Server;
  }

  /** Router constructor — callable to create new router instances. */
  export function Router(): IRouter;
  /** Router type — the interface returned by Router(). */
  export type Router = IRouter;

  interface Express {
    (): Application;
    json(options?: { limit?: string }): RequestHandler;
    static(
      root: string,
      options?: { index?: boolean; maxAge?: string; immutable?: boolean },
    ): RequestHandler;
    Router(): IRouter;
  }

  const express: Express;
  export default express;
}
