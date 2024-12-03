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
    sphere { m*<0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 1 }        
    sphere {  m*<0.4189379881297595,-2.6477949768755113e-18,4.060599922087947>, 1 }
    sphere {  m*<7.998895838423486,2.66495798843652e-18,-1.7854243240995034>, 1 }
    sphere {  m*<-4.403083962021043,8.164965809277259,-2.190920460390008>, 1}
    sphere { m*<-4.403083962021043,-8.164965809277259,-2.1909204603900116>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4189379881297595,-2.6477949768755113e-18,4.060599922087947>, <0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 0.5 }
    cylinder { m*<7.998895838423486,2.66495798843652e-18,-1.7854243240995034>, <0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 0.5}
    cylinder { m*<-4.403083962021043,8.164965809277259,-2.190920460390008>, <0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 0.5 }
    cylinder {  m*<-4.403083962021043,-8.164965809277259,-2.1909204603900116>, <0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 0.5}

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
    sphere { m*<0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 1 }        
    sphere {  m*<0.4189379881297595,-2.6477949768755113e-18,4.060599922087947>, 1 }
    sphere {  m*<7.998895838423486,2.66495798843652e-18,-1.7854243240995034>, 1 }
    sphere {  m*<-4.403083962021043,8.164965809277259,-2.190920460390008>, 1}
    sphere { m*<-4.403083962021043,-8.164965809277259,-2.1909204603900116>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.4189379881297595,-2.6477949768755113e-18,4.060599922087947>, <0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 0.5 }
    cylinder { m*<7.998895838423486,2.66495798843652e-18,-1.7854243240995034>, <0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 0.5}
    cylinder { m*<-4.403083962021043,8.164965809277259,-2.190920460390008>, <0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 0.5 }
    cylinder {  m*<-4.403083962021043,-8.164965809277259,-2.1909204603900116>, <0.3674636360012532,-5.19153940150056e-18,1.0610398846539166>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    