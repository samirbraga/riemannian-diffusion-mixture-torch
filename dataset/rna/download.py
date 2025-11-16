import aiohttp
import asyncio
import os
import itertools
from tqdm import tqdm

async def fetch_and_save(session, pdb_id, out_dir, verbose=False):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    out_file = os.path.join(out_dir, f"{pdb_id}.pdb.gz")
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            with open(out_file, "wb") as f:
                f.write(content)
            if verbose:
                print(f"Downloaded {pdb_id}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

async def download_files(in_file: str, out_dir: str):
    with open(in_file, "r") as f:
        pdb_ids = f.read().strip().split(",")

    os.makedirs(out_dir, exist_ok=True)

    chunk_size = 20
    chunked_pdb_ids = itertools.batched(pdb_ids, chunk_size)
    async with aiohttp.ClientSession() as session:
        for chunk in tqdm(chunked_pdb_ids, total=(len(pdb_ids) + chunk_size - 1) // chunk_size):
            tasks = [fetch_and_save(session, pdb_id, out_dir) for pdb_id in chunk]
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(download_files("list_file.txt", "pdb_files"))
