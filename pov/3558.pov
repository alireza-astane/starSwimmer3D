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
    sphere { m*<0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 1 }        
    sphere {  m*<0.34976272401265,0.5426895636207199,2.9226742285503793>, 1 }
    sphere {  m*<2.843736013277216,0.5160134608267689,-1.2940900680213554>, 1 }
    sphere {  m*<-1.5125877406219326,2.7424534298589953,-1.0388263079861413>, 1}
    sphere { m*<-2.63788967446595,-4.778673832471762,-1.6564270880043028>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.34976272401265,0.5426895636207199,2.9226742285503793>, <0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 0.5 }
    cylinder { m*<2.843736013277216,0.5160134608267689,-1.2940900680213554>, <0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 0.5}
    cylinder { m*<-1.5125877406219326,2.7424534298589953,-1.0388263079861413>, <0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 0.5 }
    cylinder {  m*<-2.63788967446595,-4.778673832471762,-1.6564270880043028>, <0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 0.5}

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
    sphere { m*<0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 1 }        
    sphere {  m*<0.34976272401265,0.5426895636207199,2.9226742285503793>, 1 }
    sphere {  m*<2.843736013277216,0.5160134608267689,-1.2940900680213554>, 1 }
    sphere {  m*<-1.5125877406219326,2.7424534298589953,-1.0388263079861413>, 1}
    sphere { m*<-2.63788967446595,-4.778673832471762,-1.6564270880043028>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.34976272401265,0.5426895636207199,2.9226742285503793>, <0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 0.5 }
    cylinder { m*<2.843736013277216,0.5160134608267689,-1.2940900680213554>, <0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 0.5}
    cylinder { m*<-1.5125877406219326,2.7424534298589953,-1.0388263079861413>, <0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 0.5 }
    cylinder {  m*<-2.63788967446595,-4.778673832471762,-1.6564270880043028>, <0.10902761927095844,0.4139794854403944,-0.06488054257017042>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    