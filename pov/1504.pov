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
    sphere { m*<0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 1 }        
    sphere {  m*<0.794396768758468,-1.2843508981317781e-18,3.926406867963583>, 1 }
    sphere {  m*<6.702419476068728,3.0860469226649392e-18,-1.4422858304952426>, 1 }
    sphere {  m*<-4.141896923598769,8.164965809277259,-2.235288967656917>, 1}
    sphere { m*<-4.141896923598769,-8.164965809277259,-2.2352889676569205>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.794396768758468,-1.2843508981317781e-18,3.926406867963583>, <0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 0.5 }
    cylinder { m*<6.702419476068728,3.0860469226649392e-18,-1.4422858304952426>, <0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 0.5}
    cylinder { m*<-4.141896923598769,8.164965809277259,-2.235288967656917>, <0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 0.5 }
    cylinder {  m*<-4.141896923598769,-8.164965809277259,-2.2352889676569205>, <0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 0.5}

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
    sphere { m*<0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 1 }        
    sphere {  m*<0.794396768758468,-1.2843508981317781e-18,3.926406867963583>, 1 }
    sphere {  m*<6.702419476068728,3.0860469226649392e-18,-1.4422858304952426>, 1 }
    sphere {  m*<-4.141896923598769,8.164965809277259,-2.235288967656917>, 1}
    sphere { m*<-4.141896923598769,-8.164965809277259,-2.2352889676569205>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.794396768758468,-1.2843508981317781e-18,3.926406867963583>, <0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 0.5 }
    cylinder { m*<6.702419476068728,3.0860469226649392e-18,-1.4422858304952426>, <0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 0.5}
    cylinder { m*<-4.141896923598769,8.164965809277259,-2.235288967656917>, <0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 0.5 }
    cylinder {  m*<-4.141896923598769,-8.164965809277259,-2.2352889676569205>, <0.6877116033798977,-5.328951433742996e-18,0.9283006259420892>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    