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
    sphere { m*<-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 1 }        
    sphere {  m*<0.13573730450960186,0.07735474528060246,3.3288171763139673>, 1 }
    sphere {  m*<2.576347645556426,0.022147908490012705,-1.5501884278561535>, 1 }
    sphere {  m*<-1.7799761083427212,2.2485878775222377,-1.29492466782094>, 1}
    sphere { m*<-1.5121888873048894,-2.6391040648816597,-1.1053783826583674>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13573730450960186,0.07735474528060246,3.3288171763139673>, <-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 0.5 }
    cylinder { m*<2.576347645556426,0.022147908490012705,-1.5501884278561535>, <-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 0.5}
    cylinder { m*<-1.7799761083427212,2.2485878775222377,-1.29492466782094>, <-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 0.5 }
    cylinder {  m*<-1.5121888873048894,-2.6391040648816597,-1.1053783826583674>, <-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 0.5}

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
    sphere { m*<-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 1 }        
    sphere {  m*<0.13573730450960186,0.07735474528060246,3.3288171763139673>, 1 }
    sphere {  m*<2.576347645556426,0.022147908490012705,-1.5501884278561535>, 1 }
    sphere {  m*<-1.7799761083427212,2.2485878775222377,-1.29492466782094>, 1}
    sphere { m*<-1.5121888873048894,-2.6391040648816597,-1.1053783826583674>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.13573730450960186,0.07735474528060246,3.3288171763139673>, <-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 0.5 }
    cylinder { m*<2.576347645556426,0.022147908490012705,-1.5501884278561535>, <-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 0.5}
    cylinder { m*<-1.7799761083427212,2.2485878775222377,-1.29492466782094>, <-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 0.5 }
    cylinder {  m*<-1.5121888873048894,-2.6391040648816597,-1.1053783826583674>, <-0.15836074844983114,-0.07988606689636141,-0.3209789024049701>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    