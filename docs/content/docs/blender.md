---
title: "Blender"
weight: 19
---

To generate images on Windows, you need to install Blender and load the GrainGenerator_pobedit.blend file. In the Scripting section, line 92, specify the path to the folder where images should be saved (everything before "'blender'+" and so on), in the same format, and run the script. Three samples will be created, and three images will be taken from each and saved to the specified path.

If it doesn't work, you need to open the Blender terminal (in the top left corner Window-Toggle System Console), errors will be indicated there.

To run on Linux, presumably you need to use the command:

```bash
blender yourblendfilenameorpath --python drawcar.py
```

When running on Kharisma, insert code parts marked "for generation on kharisma", and remove lines marked "for local generation"
