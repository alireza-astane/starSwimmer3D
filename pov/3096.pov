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
    sphere { m*<0.43868080600426246,1.0371415725385134,0.12611843353041619>, 1 }        
    sphere {  m*<0.6794159107459541,1.1658516507188388,3.113673204650966>, 1 }
    sphere {  m*<3.173389200010518,1.139175547924888,-1.1030910919207684>, 1 }
    sphere {  m*<-1.182934553888627,3.3656155169571154,-0.8478273318855546>, 1}
    sphere { m*<-3.767663221418406,-6.914348747201809,-2.311010751616879>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6794159107459541,1.1658516507188388,3.113673204650966>, <0.43868080600426246,1.0371415725385134,0.12611843353041619>, 0.5 }
    cylinder { m*<3.173389200010518,1.139175547924888,-1.1030910919207684>, <0.43868080600426246,1.0371415725385134,0.12611843353041619>, 0.5}
    cylinder { m*<-1.182934553888627,3.3656155169571154,-0.8478273318855546>, <0.43868080600426246,1.0371415725385134,0.12611843353041619>, 0.5 }
    cylinder {  m*<-3.767663221418406,-6.914348747201809,-2.311010751616879>, <0.43868080600426246,1.0371415725385134,0.12611843353041619>, 0.5}

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
    sphere { m*<0.43868080600426246,1.0371415725385134,0.12611843353041619>, 1 }        
    sphere {  m*<0.6794159107459541,1.1658516507188388,3.113673204650966>, 1 }
    sphere {  m*<3.173389200010518,1.139175547924888,-1.1030910919207684>, 1 }
    sphere {  m*<-1.182934553888627,3.3656155169571154,-0.8478273318855546>, 1}
    sphere { m*<-3.767663221418406,-6.914348747201809,-2.311010751616879>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6794159107459541,1.1658516507188388,3.113673204650966>, <0.43868080600426246,1.0371415725385134,0.12611843353041619>, 0.5 }
    cylinder { m*<3.173389200010518,1.139175547924888,-1.1030910919207684>, <0.43868080600426246,1.0371415725385134,0.12611843353041619>, 0.5}
    cylinder { m*<-1.182934553888627,3.3656155169571154,-0.8478273318855546>, <0.43868080600426246,1.0371415725385134,0.12611843353041619>, 0.5 }
    cylinder {  m*<-3.767663221418406,-6.914348747201809,-2.311010751616879>, <0.43868080600426246,1.0371415725385134,0.12611843353041619>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    