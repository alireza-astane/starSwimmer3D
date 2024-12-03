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
    sphere { m*<0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 1 }        
    sphere {  m*<0.5227242239953022,-1.6213022294067476e-18,4.0249215146444435>, 1 }
    sphere {  m*<7.642560959004934,2.6531851844434044e-18,-1.6937609199230248>, 1 }
    sphere {  m*<-4.3291128892670026,8.164965809277259,-2.2034892754986757>, 1}
    sphere { m*<-4.3291128892670026,-8.164965809277259,-2.2034892754986783>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5227242239953022,-1.6213022294067476e-18,4.0249215146444435>, <0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 0.5 }
    cylinder { m*<7.642560959004934,2.6531851844434044e-18,-1.6937609199230248>, <0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 0.5}
    cylinder { m*<-4.3291128892670026,8.164965809277259,-2.2034892754986757>, <0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 0.5 }
    cylinder {  m*<-4.3291128892670026,-8.164965809277259,-2.2034892754986783>, <0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 0.5}

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
    sphere { m*<0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 1 }        
    sphere {  m*<0.5227242239953022,-1.6213022294067476e-18,4.0249215146444435>, 1 }
    sphere {  m*<7.642560959004934,2.6531851844434044e-18,-1.6937609199230248>, 1 }
    sphere {  m*<-4.3291128892670026,8.164965809277259,-2.2034892754986757>, 1}
    sphere { m*<-4.3291128892670026,-8.164965809277259,-2.2034892754986783>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.5227242239953022,-1.6213022294067476e-18,4.0249215146444435>, <0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 0.5 }
    cylinder { m*<7.642560959004934,2.6531851844434044e-18,-1.6937609199230248>, <0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 0.5}
    cylinder { m*<-4.3291128892670026,8.164965809277259,-2.2034892754986757>, <0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 0.5 }
    cylinder {  m*<-4.3291128892670026,-8.164965809277259,-2.2034892754986783>, <0.45691874211074934,-5.87036495970217e-18,1.025641137755794>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    