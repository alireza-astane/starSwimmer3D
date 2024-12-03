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
    sphere { m*<-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 1 }        
    sphere {  m*<0.19155788785164551,0.10719946466099756,4.021558063727109>, 1 }
    sphere {  m*<2.5642006843263987,0.015653482464162594,-1.7009338374893819>, 1 }
    sphere {  m*<-1.7921230695727486,2.2420934514963875,-1.4456700774541684>, 1}
    sphere { m*<-1.5243358485349168,-2.64559849090751,-1.256123792291596>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19155788785164551,0.10719946466099756,4.021558063727109>, <-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 0.5 }
    cylinder { m*<2.5642006843263987,0.015653482464162594,-1.7009338374893819>, <-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 0.5}
    cylinder { m*<-1.7921230695727486,2.2420934514963875,-1.4456700774541684>, <-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 0.5 }
    cylinder {  m*<-1.5243358485349168,-2.64559849090751,-1.256123792291596>, <-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 0.5}

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
    sphere { m*<-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 1 }        
    sphere {  m*<0.19155788785164551,0.10719946466099756,4.021558063727109>, 1 }
    sphere {  m*<2.5642006843263987,0.015653482464162594,-1.7009338374893819>, 1 }
    sphere {  m*<-1.7921230695727486,2.2420934514963875,-1.4456700774541684>, 1}
    sphere { m*<-1.5243358485349168,-2.64559849090751,-1.256123792291596>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19155788785164551,0.10719946466099756,4.021558063727109>, <-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 0.5 }
    cylinder { m*<2.5642006843263987,0.015653482464162594,-1.7009338374893819>, <-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 0.5}
    cylinder { m*<-1.7921230695727486,2.2420934514963875,-1.4456700774541684>, <-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 0.5 }
    cylinder {  m*<-1.5243358485349168,-2.64559849090751,-1.256123792291596>, <-0.17050770967985843,-0.08638049292221153,-0.4717243120381984>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    