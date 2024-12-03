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
    sphere { m*<-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 1 }        
    sphere {  m*<0.16864070590428726,0.09494669295459346,3.737152777921649>, 1 }
    sphere {  m*<2.569369264598904,0.018416886545463154,-1.6367910666641095>, 1 }
    sphere {  m*<-1.7869544893002427,2.244856855577688,-1.381527306628896>, 1}
    sphere { m*<-1.5191672682624109,-2.642835086826209,-1.1919810214663233>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16864070590428726,0.09494669295459346,3.737152777921649>, <-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 0.5 }
    cylinder { m*<2.569369264598904,0.018416886545463154,-1.6367910666641095>, <-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 0.5}
    cylinder { m*<-1.7869544893002427,2.244856855577688,-1.381527306628896>, <-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 0.5 }
    cylinder {  m*<-1.5191672682624109,-2.642835086826209,-1.1919810214663233>, <-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 0.5}

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
    sphere { m*<-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 1 }        
    sphere {  m*<0.16864070590428726,0.09494669295459346,3.737152777921649>, 1 }
    sphere {  m*<2.569369264598904,0.018416886545463154,-1.6367910666641095>, 1 }
    sphere {  m*<-1.7869544893002427,2.244856855577688,-1.381527306628896>, 1}
    sphere { m*<-1.5191672682624109,-2.642835086826209,-1.1919810214663233>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.16864070590428726,0.09494669295459346,3.737152777921649>, <-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 0.5 }
    cylinder { m*<2.569369264598904,0.018416886545463154,-1.6367910666641095>, <-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 0.5}
    cylinder { m*<-1.7869544893002427,2.244856855577688,-1.381527306628896>, <-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 0.5 }
    cylinder {  m*<-1.5191672682624109,-2.642835086826209,-1.1919810214663233>, <-0.16533912940735263,-0.08361708884091096,-0.407581541212926>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    