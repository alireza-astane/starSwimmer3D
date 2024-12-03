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
    sphere { m*<1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 1 }        
    sphere {  m*<1.3691665279754914,1.4794701996224343e-18,3.6862624094493537>, 1 }
    sphere {  m*<4.627406790916099,7.373845162299714e-18,-0.8162520750641027>, 1 }
    sphere {  m*<-3.77960585498392,8.164965809277259,-2.2990390502277736>, 1}
    sphere { m*<-3.77960585498392,-8.164965809277259,-2.299039050227777>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3691665279754914,1.4794701996224343e-18,3.6862624094493537>, <1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 0.5 }
    cylinder { m*<4.627406790916099,7.373845162299714e-18,-0.8162520750641027>, <1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 0.5}
    cylinder { m*<-3.77960585498392,8.164965809277259,-2.2990390502277736>, <1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 0.5 }
    cylinder {  m*<-3.77960585498392,-8.164965809277259,-2.299039050227777>, <1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 0.5}

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
    sphere { m*<1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 1 }        
    sphere {  m*<1.3691665279754914,1.4794701996224343e-18,3.6862624094493537>, 1 }
    sphere {  m*<4.627406790916099,7.373845162299714e-18,-0.8162520750641027>, 1 }
    sphere {  m*<-3.77960585498392,8.164965809277259,-2.2990390502277736>, 1}
    sphere { m*<-3.77960585498392,-8.164965809277259,-2.299039050227777>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<1.3691665279754914,1.4794701996224343e-18,3.6862624094493537>, <1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 0.5 }
    cylinder { m*<4.627406790916099,7.373845162299714e-18,-0.8162520750641027>, <1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 0.5}
    cylinder { m*<-3.77960585498392,8.164965809277259,-2.2990390502277736>, <1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 0.5 }
    cylinder {  m*<-3.77960585498392,-8.164965809277259,-2.299039050227777>, <1.1576949191322823,1.5265807662157175e-19,0.6937164457691763>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    