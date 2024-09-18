# Snakefile
CUSTOM_HUCIDS = ['1805000203']

DEMO_HUCIDS = ['1801010902', '1605030102']
CA_HUC8_HUCIDS= ['1801010902', '1801010805']

rule all:
    input: 
        expand("data/{hucid}/{hucid}-floors.tif", hucid=CA_HUC8_HUCIDS)

rule demo_all:
    input:
        expand("data/{hucid}/{hucid}-floors.tif", hucid=DEMO_HUCIDS)

rule custom_all:
    input:
        expand("data/{hucid}/{hucid}-floors.tif", hucid=CUSTOM_HUCIDS)

rule download_data:
    params:
        hucid = '{hucid}'
    output:
        "data/{hucid}/{hucid}-dem_raw.tif",
	"data/{hucid}/{hucid}-flowlines_raw.shp"
    shell:
        "poetry run python src/download_huc.py {params.hucid} {output}"

rule preprocess_dem:
      input: 
          dem_path = "data/{hucid}/{hucid}-dem_raw.tif",
          land_shapefile = "california_mask/California.shp"
      output:
          "data/{hucid}/{hucid}-dem.tif"
      shell:
          "poetry run python src/ocean_mask.py {input.dem_path} {input.land_shapefile} {output}"

rule preprocess_nhd:
     input:
         "data/{hucid}/{hucid}-flowlines_raw.shp"
     output:
         "data/{hucid}/{hucid}-flowlines.shp"
     shell:
         "poetry run python src/filter_nhd.py {input} {output}"

rule extract_valleys:
     input:
         dem = "data/{hucid}/{hucid}-dem.tif",
         network = "data/{hucid}/{hucid}-flowlines.shp"
     params:
         param = "configs/params.toml",
         wbt = os.path.expanduser("~/opt/WBT")
     output:
         terrain_dir = directory("data/{hucid}/derived/"),
         ofile = directory("data/{hucid}/{hucid}-floors.tif/"),
     shell:
         "poetry run python -m valleyfloor --dem {input.dem} --flowlines {input.network} --param {params.param}  --wbt {params.wbt} --terrain_dir {output.terrain_dir} --ofile {output.ofile}" 

