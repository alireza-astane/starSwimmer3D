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
    sphere { m*<-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 1 }        
    sphere {  m*<0.19393046780311457,0.2481110249478765,2.832386002689712>, 1 }
    sphere {  m*<2.6879037570676854,0.22143492215392568,-1.3843782938820253>, 1 }
    sphere {  m*<-1.6684199968314686,2.447874891186154,-1.1291145338468103>, 1}
    sphere { m*<-2.010306714324687,-3.592318222977114,-1.2928094812916733>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19393046780311457,0.2481110249478765,2.832386002689712>, <-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 0.5 }
    cylinder { m*<2.6879037570676854,0.22143492215392568,-1.3843782938820253>, <-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 0.5}
    cylinder { m*<-1.6684199968314686,2.447874891186154,-1.1291145338468103>, <-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 0.5 }
    cylinder {  m*<-2.010306714324687,-3.592318222977114,-1.2928094812916733>, <-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 0.5}

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
    sphere { m*<-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 1 }        
    sphere {  m*<0.19393046780311457,0.2481110249478765,2.832386002689712>, 1 }
    sphere {  m*<2.6879037570676854,0.22143492215392568,-1.3843782938820253>, 1 }
    sphere {  m*<-1.6684199968314686,2.447874891186154,-1.1291145338468103>, 1}
    sphere { m*<-2.010306714324687,-3.592318222977114,-1.2928094812916733>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.19393046780311457,0.2481110249478765,2.832386002689712>, <-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 0.5 }
    cylinder { m*<2.6879037570676854,0.22143492215392568,-1.3843782938820253>, <-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 0.5}
    cylinder { m*<-1.6684199968314686,2.447874891186154,-1.1291145338468103>, <-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 0.5 }
    cylinder {  m*<-2.010306714324687,-3.592318222977114,-1.2928094812916733>, <-0.04680463693857695,0.11940094676755098,-0.15516876843083827>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    