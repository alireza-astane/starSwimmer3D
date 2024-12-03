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
    sphere { m*<0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 1 }        
    sphere {  m*<1.160480939978754,9.198229437602316e-19,3.7793159842709896>, 1 }
    sphere {  m*<5.403557768071915,5.295955000464277e-18,-1.064692984685694>, 1 }
    sphere {  m*<-3.9055728085622605,8.164965809277259,-2.2761532576217594>, 1}
    sphere { m*<-3.9055728085622605,-8.164965809277259,-2.276153257621763>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.160480939978754,9.198229437602316e-19,3.7793159842709896>, <0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 0.5 }
    cylinder { m*<5.403557768071915,5.295955000464277e-18,-1.064692984685694>, <0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 0.5}
    cylinder { m*<-3.9055728085622605,8.164965809277259,-2.2761532576217594>, <0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 0.5 }
    cylinder {  m*<-3.9055728085622605,-8.164965809277259,-2.276153257621763>, <0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 0.5}

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
    sphere { m*<0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 1 }        
    sphere {  m*<1.160480939978754,9.198229437602316e-19,3.7793159842709896>, 1 }
    sphere {  m*<5.403557768071915,5.295955000464277e-18,-1.064692984685694>, 1 }
    sphere {  m*<-3.9055728085622605,8.164965809277259,-2.2761532576217594>, 1}
    sphere { m*<-3.9055728085622605,-8.164965809277259,-2.276153257621763>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.160480939978754,9.198229437602316e-19,3.7793159842709896>, <0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 0.5 }
    cylinder { m*<5.403557768071915,5.295955000464277e-18,-1.064692984685694>, <0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 0.5}
    cylinder { m*<-3.9055728085622605,8.164965809277259,-2.2761532576217594>, <0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 0.5 }
    cylinder {  m*<-3.9055728085622605,-8.164965809277259,-2.276153257621763>, <0.9901426415065497,-1.6428277524638154e-18,0.7841491160634004>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    