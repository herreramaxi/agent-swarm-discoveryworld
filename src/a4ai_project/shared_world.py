from collections import defaultdict
from pathlib import Path
from random import Random

import pygame
from discoveryworld.Layer import Layer


def randomize_start_positions(api, spawn_seed: int) -> None:
    """Place each user agent on a distinct passable landmark."""
    world = api.world
    agents = world.getUserAgents()
    occupied = {agent.getWorldLocation() for agent in agents + world.getNPCAgents()}
    landmarks = {}
    for name, location in sorted(world.getTeleportLocations().items()):
        position = (location["gridX"], location["gridY"])
        if position not in occupied and world.isPassable(*position)[0]:
            landmarks.setdefault(position, name)
    if len(landmarks) < len(agents):
        raise ValueError(f"Need {len(agents)} unoccupied landmarks, found {len(landmarks)}.")

    positions = Random(spawn_seed).sample(list(landmarks), len(agents))
    for agent_idx, (agent, (x, y)) in enumerate(zip(agents, positions)):
        world.removeObject(agent)
        world.addObject(x, y, Layer.AGENT, agent)
        print(f"Agent {agent_idx} spawn: {landmarks[(x, y)]} at ({x}, {y})")


def save_shared_frame(api) -> Path:
    """Save the whole map with each user agent's index drawn on its tile."""
    world = api.world
    tile_size = 32
    surface = pygame.Surface((world.sizeX * tile_size, world.sizeY * tile_size))
    world.renderViewport(surface, 0, 0, world.sizeX, world.sizeY, 0, 0)

    ids_by_tile = defaultdict(list)
    for agent_idx, agent in enumerate(world.getUserAgents()):
        ids_by_tile[agent.getWorldLocation()].append(str(agent_idx))
    font = pygame.font.Font(None, 20)
    for (x, y), agent_ids in ids_by_tile.items():
        tile = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
        pygame.draw.rect(surface, (255, 223, 94), tile, width=2)
        label = font.render("/".join(agent_ids), True, (255, 223, 94), (25, 25, 25))
        label_rect = label.get_rect(midbottom=tile.midtop if y else tile.midbottom)
        surface.blit(label, label_rect.clamp(surface.get_rect()))

    path = Path(api.FRAME_DIR) / f"shared_world_frame_{world.getStepCounter():04d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))
    return path
