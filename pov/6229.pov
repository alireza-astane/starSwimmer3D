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
    sphere { m*<-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 1 }        
    sphere {  m*<0.013571886382601761,0.1249404434035769,8.927575998306184>, 1 }
    sphere {  m*<7.368923324382574,0.03602016740921968,-5.651917291739176>, 1 }
    sphere {  m*<-4.101277804443482,3.0797390421317563,-2.311997260922716>, 1}
    sphere { m*<-2.7961856588338394,-3.0262008185051,-1.616236840785887>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.013571886382601761,0.1249404434035769,8.927575998306184>, <-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 0.5 }
    cylinder { m*<7.368923324382574,0.03602016740921968,-5.651917291739176>, <-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 0.5}
    cylinder { m*<-4.101277804443482,3.0797390421317563,-2.311997260922716>, <-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 0.5 }
    cylinder {  m*<-2.7961856588338394,-3.0262008185051,-1.616236840785887>, <-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 0.5}

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
    sphere { m*<-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 1 }        
    sphere {  m*<0.013571886382601761,0.1249404434035769,8.927575998306184>, 1 }
    sphere {  m*<7.368923324382574,0.03602016740921968,-5.651917291739176>, 1 }
    sphere {  m*<-4.101277804443482,3.0797390421317563,-2.311997260922716>, 1}
    sphere { m*<-2.7961856588338394,-3.0262008185051,-1.616236840785887>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.013571886382601761,0.1249404434035769,8.927575998306184>, <-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 0.5 }
    cylinder { m*<7.368923324382574,0.03602016740921968,-5.651917291739176>, <-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 0.5}
    cylinder { m*<-4.101277804443482,3.0797390421317563,-2.311997260922716>, <-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 0.5 }
    cylinder {  m*<-2.7961856588338394,-3.0262008185051,-1.616236840785887>, <-1.4441648589950475,-0.43266482593951416,-0.9499619538934632>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    