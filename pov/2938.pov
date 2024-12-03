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
    sphere { m*<0.5543256070599384,1.1209363854258836,0.19362470280148114>, 1 }        
    sphere {  m*<0.7956306434165827,1.2297100848598224,3.18192105786187>, 1 }
    sphere {  m*<3.2888778324791166,1.229710084859822,-1.0353611506287455>, 1 }
    sphere {  m*<-1.2968952900074049,3.7891530607913055,-0.9009257384907225>, 1}
    sphere { m*<-3.963849909733946,-7.3904996049714295,-2.4771492400712907>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7956306434165827,1.2297100848598224,3.18192105786187>, <0.5543256070599384,1.1209363854258836,0.19362470280148114>, 0.5 }
    cylinder { m*<3.2888778324791166,1.229710084859822,-1.0353611506287455>, <0.5543256070599384,1.1209363854258836,0.19362470280148114>, 0.5}
    cylinder { m*<-1.2968952900074049,3.7891530607913055,-0.9009257384907225>, <0.5543256070599384,1.1209363854258836,0.19362470280148114>, 0.5 }
    cylinder {  m*<-3.963849909733946,-7.3904996049714295,-2.4771492400712907>, <0.5543256070599384,1.1209363854258836,0.19362470280148114>, 0.5}

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
    sphere { m*<0.5543256070599384,1.1209363854258836,0.19362470280148114>, 1 }        
    sphere {  m*<0.7956306434165827,1.2297100848598224,3.18192105786187>, 1 }
    sphere {  m*<3.2888778324791166,1.229710084859822,-1.0353611506287455>, 1 }
    sphere {  m*<-1.2968952900074049,3.7891530607913055,-0.9009257384907225>, 1}
    sphere { m*<-3.963849909733946,-7.3904996049714295,-2.4771492400712907>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7956306434165827,1.2297100848598224,3.18192105786187>, <0.5543256070599384,1.1209363854258836,0.19362470280148114>, 0.5 }
    cylinder { m*<3.2888778324791166,1.229710084859822,-1.0353611506287455>, <0.5543256070599384,1.1209363854258836,0.19362470280148114>, 0.5}
    cylinder { m*<-1.2968952900074049,3.7891530607913055,-0.9009257384907225>, <0.5543256070599384,1.1209363854258836,0.19362470280148114>, 0.5 }
    cylinder {  m*<-3.963849909733946,-7.3904996049714295,-2.4771492400712907>, <0.5543256070599384,1.1209363854258836,0.19362470280148114>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    