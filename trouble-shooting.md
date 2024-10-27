## Trouble-Shooting

### On MacOS Version with `untangle`

error message:

```shell
  File ".../Documents/simulators/vlm-tamp/pybullet_planning/world_builder/world_utils.py", line 157, in read_xml
    content = untangle.parse(plan_path).svg.g.g.g
  File ".../miniconda3/envs/cogarch/lib/python3.8/site-packages/untangle.py", line 205, in parse
    parser.parse(filename)
  File ".../miniconda3/envs/cogarch/lib/python3.8/xml/sax/expatreader.py", line 111, in parse
    xmlreader.IncrementalParser.parse(self, source)
  File ".../miniconda3/envs/cogarch/lib/python3.8/xml/sax/xmlreader.py", line 125, in parse
    self.feed(buffer)
  File ".../miniconda3/envs/cogarch/lib/python3.8/xml/sax/expatreader.py", line 217, in feed
    self._parser.Parse(data, isFinal)
  File "/private/var/folders/nz/j6p8yfhx1mv_0grj5xl4650h0000gp/T/abs_40bvsc0ovr/croot/python-split_1710966196798/work/Modules/pyexpat.c", line 668, in ExternalEntityRef

loading floor plan kitchen_v2.svg...
  File ".../miniconda3/envs/cogarch/lib/python3.8/site-packages/defusedxml/expatreader.py", line 46, in defused_external_entity_ref_handler
    raise ExternalReferenceForbidden(context, base, sysid, pubid)
defusedxml.common.ExternalReferenceForbidden: ExternalReferenceForbidden(system_id='http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd', public_id=-//W3C//DTD SVG 1.1//EN)
```

solution:

```shell
pip uninstall untangle
pip install untangle==1.1.1
```

### Pybullet error 

```shell
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
Failed to create an OpenGL context
```

solution in [StackOverflow](https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris): add the following to `~/.bashrc` or `~/.zshrc` (if using zsh).

```shell
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```