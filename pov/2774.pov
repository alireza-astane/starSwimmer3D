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
    sphere { m*<0.6757903158203712,0.9576814449448637,0.2654412666476851>, 1 }        
    sphere {  m*<0.9180769733875629,1.0464926729795678,3.2543187996924194>, 1 }
    sphere {  m*<3.4113241624500974,1.0464926729795674,-0.9629634087981964>, 1 }
    sphere {  m*<-1.7599350301152485,4.55266480878157,-1.1747060811914947>, 1}
    sphere { m*<-3.9266576929372783,-7.492585978296157,-2.455156964476034>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9180769733875629,1.0464926729795678,3.2543187996924194>, <0.6757903158203712,0.9576814449448637,0.2654412666476851>, 0.5 }
    cylinder { m*<3.4113241624500974,1.0464926729795674,-0.9629634087981964>, <0.6757903158203712,0.9576814449448637,0.2654412666476851>, 0.5}
    cylinder { m*<-1.7599350301152485,4.55266480878157,-1.1747060811914947>, <0.6757903158203712,0.9576814449448637,0.2654412666476851>, 0.5 }
    cylinder {  m*<-3.9266576929372783,-7.492585978296157,-2.455156964476034>, <0.6757903158203712,0.9576814449448637,0.2654412666476851>, 0.5}

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
    sphere { m*<0.6757903158203712,0.9576814449448637,0.2654412666476851>, 1 }        
    sphere {  m*<0.9180769733875629,1.0464926729795678,3.2543187996924194>, 1 }
    sphere {  m*<3.4113241624500974,1.0464926729795674,-0.9629634087981964>, 1 }
    sphere {  m*<-1.7599350301152485,4.55266480878157,-1.1747060811914947>, 1}
    sphere { m*<-3.9266576929372783,-7.492585978296157,-2.455156964476034>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.9180769733875629,1.0464926729795678,3.2543187996924194>, <0.6757903158203712,0.9576814449448637,0.2654412666476851>, 0.5 }
    cylinder { m*<3.4113241624500974,1.0464926729795674,-0.9629634087981964>, <0.6757903158203712,0.9576814449448637,0.2654412666476851>, 0.5}
    cylinder { m*<-1.7599350301152485,4.55266480878157,-1.1747060811914947>, <0.6757903158203712,0.9576814449448637,0.2654412666476851>, 0.5 }
    cylinder {  m*<-3.9266576929372783,-7.492585978296157,-2.455156964476034>, <0.6757903158203712,0.9576814449448637,0.2654412666476851>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    