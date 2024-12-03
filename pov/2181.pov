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
    sphere { m*<1.138884438273639,0.24318976642219464,0.5392516829469864>, 1 }        
    sphere {  m*<1.383046492297334,0.26174900318046534,3.529240922675558>, 1 }
    sphere {  m*<3.876293681359871,0.26174900318046534,-0.6880412858150604>, 1 }
    sphere {  m*<-3.2632697903403067,7.313500684985272,-2.0635909921562314>, 1}
    sphere { m*<-3.754872061215639,-7.983164739424665,-2.3535769213177>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.383046492297334,0.26174900318046534,3.529240922675558>, <1.138884438273639,0.24318976642219464,0.5392516829469864>, 0.5 }
    cylinder { m*<3.876293681359871,0.26174900318046534,-0.6880412858150604>, <1.138884438273639,0.24318976642219464,0.5392516829469864>, 0.5}
    cylinder { m*<-3.2632697903403067,7.313500684985272,-2.0635909921562314>, <1.138884438273639,0.24318976642219464,0.5392516829469864>, 0.5 }
    cylinder {  m*<-3.754872061215639,-7.983164739424665,-2.3535769213177>, <1.138884438273639,0.24318976642219464,0.5392516829469864>, 0.5}

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
    sphere { m*<1.138884438273639,0.24318976642219464,0.5392516829469864>, 1 }        
    sphere {  m*<1.383046492297334,0.26174900318046534,3.529240922675558>, 1 }
    sphere {  m*<3.876293681359871,0.26174900318046534,-0.6880412858150604>, 1 }
    sphere {  m*<-3.2632697903403067,7.313500684985272,-2.0635909921562314>, 1}
    sphere { m*<-3.754872061215639,-7.983164739424665,-2.3535769213177>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.383046492297334,0.26174900318046534,3.529240922675558>, <1.138884438273639,0.24318976642219464,0.5392516829469864>, 0.5 }
    cylinder { m*<3.876293681359871,0.26174900318046534,-0.6880412858150604>, <1.138884438273639,0.24318976642219464,0.5392516829469864>, 0.5}
    cylinder { m*<-3.2632697903403067,7.313500684985272,-2.0635909921562314>, <1.138884438273639,0.24318976642219464,0.5392516829469864>, 0.5 }
    cylinder {  m*<-3.754872061215639,-7.983164739424665,-2.3535769213177>, <1.138884438273639,0.24318976642219464,0.5392516829469864>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    