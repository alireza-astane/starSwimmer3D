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
    sphere { m*<1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 1 }        
    sphere {  m*<1.512881552133779,-1.2111243361510393e-19,3.6175799645214024>, 1 }
    sphere {  m*<4.060567795240052,5.918151222702914e-18,-0.620652060839441>, 1 }
    sphere {  m*<-3.696550707233158,8.164965809277259,-2.3150053279628073>, 1}
    sphere { m*<-3.696550707233158,-8.164965809277259,-2.315005327962811>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.512881552133779,-1.2111243361510393e-19,3.6175799645214024>, <1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 0.5 }
    cylinder { m*<4.060567795240052,5.918151222702914e-18,-0.620652060839441>, <1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 0.5}
    cylinder { m*<-3.696550707233158,8.164965809277259,-2.3150053279628073>, <1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 0.5 }
    cylinder {  m*<-3.696550707233158,-8.164965809277259,-2.315005327962811>, <1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 0.5}

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
    sphere { m*<1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 1 }        
    sphere {  m*<1.512881552133779,-1.2111243361510393e-19,3.6175799645214024>, 1 }
    sphere {  m*<4.060567795240052,5.918151222702914e-18,-0.620652060839441>, 1 }
    sphere {  m*<-3.696550707233158,8.164965809277259,-2.3150053279628073>, 1}
    sphere { m*<-3.696550707233158,-8.164965809277259,-2.315005327962811>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.512881552133779,-1.2111243361510393e-19,3.6175799645214024>, <1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 0.5 }
    cylinder { m*<4.060567795240052,5.918151222702914e-18,-0.620652060839441>, <1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 0.5}
    cylinder { m*<-3.696550707233158,8.164965809277259,-2.3150053279628073>, <1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 0.5 }
    cylinder {  m*<-3.696550707233158,-8.164965809277259,-2.315005327962811>, <1.2709598335940693,-7.474564228474865e-19,0.6273401020860068>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    