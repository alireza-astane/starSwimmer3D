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
    sphere { m*<0.5772971096342122,1.0908104239876926,0.20720667688638233>, 1 }        
    sphere {  m*<0.8188038787471241,1.1957930759533213,3.195622456508465>, 1 }
    sphere {  m*<3.3120510678096577,1.1957930759533208,-1.02165975198215>, 1 }
    sphere {  m*<-1.3895427130312097,3.9384550561550338,-0.9557050004757919>, 1}
    sphere { m*<-3.9569335340305645,-7.409092992836069,-2.4730595073890766>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8188038787471241,1.1957930759533213,3.195622456508465>, <0.5772971096342122,1.0908104239876926,0.20720667688638233>, 0.5 }
    cylinder { m*<3.3120510678096577,1.1957930759533208,-1.02165975198215>, <0.5772971096342122,1.0908104239876926,0.20720667688638233>, 0.5}
    cylinder { m*<-1.3895427130312097,3.9384550561550338,-0.9557050004757919>, <0.5772971096342122,1.0908104239876926,0.20720667688638233>, 0.5 }
    cylinder {  m*<-3.9569335340305645,-7.409092992836069,-2.4730595073890766>, <0.5772971096342122,1.0908104239876926,0.20720667688638233>, 0.5}

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
    sphere { m*<0.5772971096342122,1.0908104239876926,0.20720667688638233>, 1 }        
    sphere {  m*<0.8188038787471241,1.1957930759533213,3.195622456508465>, 1 }
    sphere {  m*<3.3120510678096577,1.1957930759533208,-1.02165975198215>, 1 }
    sphere {  m*<-1.3895427130312097,3.9384550561550338,-0.9557050004757919>, 1}
    sphere { m*<-3.9569335340305645,-7.409092992836069,-2.4730595073890766>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8188038787471241,1.1957930759533213,3.195622456508465>, <0.5772971096342122,1.0908104239876926,0.20720667688638233>, 0.5 }
    cylinder { m*<3.3120510678096577,1.1957930759533208,-1.02165975198215>, <0.5772971096342122,1.0908104239876926,0.20720667688638233>, 0.5}
    cylinder { m*<-1.3895427130312097,3.9384550561550338,-0.9557050004757919>, <0.5772971096342122,1.0908104239876926,0.20720667688638233>, 0.5 }
    cylinder {  m*<-3.9569335340305645,-7.409092992836069,-2.4730595073890766>, <0.5772971096342122,1.0908104239876926,0.20720667688638233>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    