#version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 1 }        
    sphere {  m*<0.0999692711108757,0.058231209558438884,2.8849311093083965>, 1 }
    sphere {  m*<2.5832699707318927,0.02584895996516008,-1.4642814487830145>, 1 }
    sphere {  m*<-1.7730537831672541,2.252288928997385,-1.209017688747801>, 1}
    sphere { m*<-1.5052665621294223,-2.6354030134065125,-1.0194714035852281>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0999692711108757,0.058231209558438884,2.8849311093083965>, <-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 0.5 }
    cylinder { m*<2.5832699707318927,0.02584895996516008,-1.4642814487830145>, <-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 0.5}
    cylinder { m*<-1.7730537831672541,2.252288928997385,-1.209017688747801>, <-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 0.5 }
    cylinder {  m*<-1.5052665621294223,-2.6354030134065125,-1.0194714035852281>, <-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    #version 3.7; 
    global_settings { assumed_gamma 1.0 }
    

    camera {
    location  <20, 20, 20>
    right     x*image_width/image_height
    look_at   <0, 0, 0>
    angle 58
    }

    background { color rgb<1,1,1>*0.03 }


    light_source { <-20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    light_source { < 20, 30, -25> color red 0.6 green 0.6 blue 0.6 }
    
    #declare m = 1;
    union {

    union {
    sphere { m*<-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 1 }        
    sphere {  m*<0.0999692711108757,0.058231209558438884,2.8849311093083965>, 1 }
    sphere {  m*<2.5832699707318927,0.02584895996516008,-1.4642814487830145>, 1 }
    sphere {  m*<-1.7730537831672541,2.252288928997385,-1.209017688747801>, 1}
    sphere { m*<-1.5052665621294223,-2.6354030134065125,-1.0194714035852281>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.0999692711108757,0.058231209558438884,2.8849311093083965>, <-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 0.5 }
    cylinder { m*<2.5832699707318927,0.02584895996516008,-1.4642814487830145>, <-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 0.5}
    cylinder { m*<-1.7730537831672541,2.252288928997385,-1.209017688747801>, <-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 0.5 }
    cylinder {  m*<-1.5052665621294223,-2.6354030134065125,-1.0194714035852281>, <-0.15143842327436408,-0.07618501542121403,-0.23507192333183094>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    