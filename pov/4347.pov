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
    sphere { m*<-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 1 }        
    sphere {  m*<0.2512199679506127,0.13909805710840384,4.761972438464441>, 1 }
    sphere {  m*<2.549716352164925,0.007909370832028847,-1.8806863322571763>, 1 }
    sphere {  m*<-1.8066074017342222,2.234349339864254,-1.625422572221963>, 1}
    sphere { m*<-1.5388201806963904,-2.6533426025396434,-1.4358762870593904>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2512199679506127,0.13909805710840384,4.761972438464441>, <-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 0.5 }
    cylinder { m*<2.549716352164925,0.007909370832028847,-1.8806863322571763>, <-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 0.5}
    cylinder { m*<-1.8066074017342222,2.234349339864254,-1.625422572221963>, <-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 0.5 }
    cylinder {  m*<-1.5388201806963904,-2.6533426025396434,-1.4358762870593904>, <-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 0.5}

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
    sphere { m*<-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 1 }        
    sphere {  m*<0.2512199679506127,0.13909805710840384,4.761972438464441>, 1 }
    sphere {  m*<2.549716352164925,0.007909370832028847,-1.8806863322571763>, 1 }
    sphere {  m*<-1.8066074017342222,2.234349339864254,-1.625422572221963>, 1}
    sphere { m*<-1.5388201806963904,-2.6533426025396434,-1.4358762870593904>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.2512199679506127,0.13909805710840384,4.761972438464441>, <-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 0.5 }
    cylinder { m*<2.549716352164925,0.007909370832028847,-1.8806863322571763>, <-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 0.5}
    cylinder { m*<-1.8066074017342222,2.234349339864254,-1.625422572221963>, <-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 0.5 }
    cylinder {  m*<-1.5388201806963904,-2.6533426025396434,-1.4358762870593904>, <-0.18499204184133217,-0.09412460455434533,-0.6514768068059944>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    