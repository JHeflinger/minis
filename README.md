# minis

minis are a bunch of common, easy to use third party libraries, organized and configured to be fitted for [tiny](https://github.com/JHeflinger/tiny)! This makes it super easy to include them and use them with a single-line configuration in your projects.
 
## How To Use

Once you've gotten started with [tiny](https://github.com/JHeflinger/tiny), just add in the subfolders as a module via your `.tinyconf` configuration file. For example, if you want to add raylib to your project, just add the following line to your `.tinyconf`:

```
MODULE raylib https://github.com/JHeflinger/minis.git raylib
```

The **first argument** is the name for the module, this is only really used for bookkeeping within tiny. This can really be anything as long as it is unique within your project!

The **second argument** is the link to the github repository, which is the github link to this repo!

The **last argument** is the path to the module folder, which in this case is the `raylib/` folder within this repository. This way tiny will only check out the raylib module in particular, and use the `.tinymodule` configuration for raylib alone.
