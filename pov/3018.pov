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
    sphere { m*<0.49752273703479877,1.1483738069762703,0.16021107704460164>, 1 }        
    sphere {  m*<0.7382578417764902,1.2770838851565958,3.1477658481651507>, 1 }
    sphere {  m*<3.232231131041056,1.250407782362645,-1.0689984484065826>, 1 }
    sphere {  m*<-1.12409262285809,3.47684775139487,-0.8137346883713689>, 1}
    sphere { m*<-3.9552180116594897,-7.268894188397378,-2.41967881350881>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7382578417764902,1.2770838851565958,3.1477658481651507>, <0.49752273703479877,1.1483738069762703,0.16021107704460164>, 0.5 }
    cylinder { m*<3.232231131041056,1.250407782362645,-1.0689984484065826>, <0.49752273703479877,1.1483738069762703,0.16021107704460164>, 0.5}
    cylinder { m*<-1.12409262285809,3.47684775139487,-0.8137346883713689>, <0.49752273703479877,1.1483738069762703,0.16021107704460164>, 0.5 }
    cylinder {  m*<-3.9552180116594897,-7.268894188397378,-2.41967881350881>, <0.49752273703479877,1.1483738069762703,0.16021107704460164>, 0.5}

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
    sphere { m*<0.49752273703479877,1.1483738069762703,0.16021107704460164>, 1 }        
    sphere {  m*<0.7382578417764902,1.2770838851565958,3.1477658481651507>, 1 }
    sphere {  m*<3.232231131041056,1.250407782362645,-1.0689984484065826>, 1 }
    sphere {  m*<-1.12409262285809,3.47684775139487,-0.8137346883713689>, 1}
    sphere { m*<-3.9552180116594897,-7.268894188397378,-2.41967881350881>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7382578417764902,1.2770838851565958,3.1477658481651507>, <0.49752273703479877,1.1483738069762703,0.16021107704460164>, 0.5 }
    cylinder { m*<3.232231131041056,1.250407782362645,-1.0689984484065826>, <0.49752273703479877,1.1483738069762703,0.16021107704460164>, 0.5}
    cylinder { m*<-1.12409262285809,3.47684775139487,-0.8137346883713689>, <0.49752273703479877,1.1483738069762703,0.16021107704460164>, 0.5 }
    cylinder {  m*<-3.9552180116594897,-7.268894188397378,-2.41967881350881>, <0.49752273703479877,1.1483738069762703,0.16021107704460164>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    