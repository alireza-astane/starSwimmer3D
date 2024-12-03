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
    sphere { m*<0.586022003072534,1.0792798520957467,0.21236530167175344>, 1 }        
    sphere {  m*<0.8276034623286562,1.1828235466772512,3.2008252989010266>, 1 }
    sphere {  m*<3.32085065139119,1.1828235466772508,-1.0164569095895883>, 1 }
    sphere {  m*<-1.4239774007276167,3.9943885171739306,-0.9760650758148439>, 1}
    sphere { m*<-3.9542963708963907,-7.4162358761789005,-2.4715001201667626>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8276034623286562,1.1828235466772512,3.2008252989010266>, <0.586022003072534,1.0792798520957467,0.21236530167175344>, 0.5 }
    cylinder { m*<3.32085065139119,1.1828235466772508,-1.0164569095895883>, <0.586022003072534,1.0792798520957467,0.21236530167175344>, 0.5}
    cylinder { m*<-1.4239774007276167,3.9943885171739306,-0.9760650758148439>, <0.586022003072534,1.0792798520957467,0.21236530167175344>, 0.5 }
    cylinder {  m*<-3.9542963708963907,-7.4162358761789005,-2.4715001201667626>, <0.586022003072534,1.0792798520957467,0.21236530167175344>, 0.5}

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
    sphere { m*<0.586022003072534,1.0792798520957467,0.21236530167175344>, 1 }        
    sphere {  m*<0.8276034623286562,1.1828235466772512,3.2008252989010266>, 1 }
    sphere {  m*<3.32085065139119,1.1828235466772508,-1.0164569095895883>, 1 }
    sphere {  m*<-1.4239774007276167,3.9943885171739306,-0.9760650758148439>, 1}
    sphere { m*<-3.9542963708963907,-7.4162358761789005,-2.4715001201667626>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8276034623286562,1.1828235466772512,3.2008252989010266>, <0.586022003072534,1.0792798520957467,0.21236530167175344>, 0.5 }
    cylinder { m*<3.32085065139119,1.1828235466772508,-1.0164569095895883>, <0.586022003072534,1.0792798520957467,0.21236530167175344>, 0.5}
    cylinder { m*<-1.4239774007276167,3.9943885171739306,-0.9760650758148439>, <0.586022003072534,1.0792798520957467,0.21236530167175344>, 0.5 }
    cylinder {  m*<-3.9542963708963907,-7.4162358761789005,-2.4715001201667626>, <0.586022003072534,1.0792798520957467,0.21236530167175344>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    