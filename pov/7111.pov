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
    sphere { m*<-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 1 }        
    sphere {  m*<0.6626313331781278,-0.1781893575433775,9.249383614338752>, 1 }
    sphere {  m*<8.030418531500931,-0.4632816083356398,-5.321293814735186>, 1 }
    sphere {  m*<-6.865544662188065,6.059799765285015,-3.83048691155358>, 1}
    sphere { m*<-2.306883979636844,-4.544483102838251,-1.317853254691474>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6626313331781278,-0.1781893575433775,9.249383614338752>, <-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 0.5 }
    cylinder { m*<8.030418531500931,-0.4632816083356398,-5.321293814735186>, <-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 0.5}
    cylinder { m*<-6.865544662188065,6.059799765285015,-3.83048691155358>, <-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 0.5 }
    cylinder {  m*<-2.306883979636844,-4.544483102838251,-1.317853254691474>, <-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 0.5}

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
    sphere { m*<-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 1 }        
    sphere {  m*<0.6626313331781278,-0.1781893575433775,9.249383614338752>, 1 }
    sphere {  m*<8.030418531500931,-0.4632816083356398,-5.321293814735186>, 1 }
    sphere {  m*<-6.865544662188065,6.059799765285015,-3.83048691155358>, 1}
    sphere { m*<-2.306883979636844,-4.544483102838251,-1.317853254691474>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.6626313331781278,-0.1781893575433775,9.249383614338752>, <-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 0.5 }
    cylinder { m*<8.030418531500931,-0.4632816083356398,-5.321293814735186>, <-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 0.5}
    cylinder { m*<-6.865544662188065,6.059799765285015,-3.83048691155358>, <-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 0.5 }
    cylinder {  m*<-2.306883979636844,-4.544483102838251,-1.317853254691474>, <-0.7565361610220349,-1.1681282714232952,-0.5999064826964021>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    