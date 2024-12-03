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
    sphere { m*<0.5221807715530196,1.1625541602036464,0.17461898921340124>, 1 }        
    sphere {  m*<0.7631914069776926,1.2766335653979946,3.162741070327294>, 1 }
    sphere {  m*<3.256438596040229,1.2766335653979939,-1.0545411381633218>, 1 }
    sphere {  m*<-1.161496182041386,3.5740649663858877,-0.8208689947366055>, 1}
    sphere { m*<-3.9734847967676035,-7.364955980855061,-2.4828464446724636>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7631914069776926,1.2766335653979946,3.162741070327294>, <0.5221807715530196,1.1625541602036464,0.17461898921340124>, 0.5 }
    cylinder { m*<3.256438596040229,1.2766335653979939,-1.0545411381633218>, <0.5221807715530196,1.1625541602036464,0.17461898921340124>, 0.5}
    cylinder { m*<-1.161496182041386,3.5740649663858877,-0.8208689947366055>, <0.5221807715530196,1.1625541602036464,0.17461898921340124>, 0.5 }
    cylinder {  m*<-3.9734847967676035,-7.364955980855061,-2.4828464446724636>, <0.5221807715530196,1.1625541602036464,0.17461898921340124>, 0.5}

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
    sphere { m*<0.5221807715530196,1.1625541602036464,0.17461898921340124>, 1 }        
    sphere {  m*<0.7631914069776926,1.2766335653979946,3.162741070327294>, 1 }
    sphere {  m*<3.256438596040229,1.2766335653979939,-1.0545411381633218>, 1 }
    sphere {  m*<-1.161496182041386,3.5740649663858877,-0.8208689947366055>, 1}
    sphere { m*<-3.9734847967676035,-7.364955980855061,-2.4828464446724636>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.7631914069776926,1.2766335653979946,3.162741070327294>, <0.5221807715530196,1.1625541602036464,0.17461898921340124>, 0.5 }
    cylinder { m*<3.256438596040229,1.2766335653979939,-1.0545411381633218>, <0.5221807715530196,1.1625541602036464,0.17461898921340124>, 0.5}
    cylinder { m*<-1.161496182041386,3.5740649663858877,-0.8208689947366055>, <0.5221807715530196,1.1625541602036464,0.17461898921340124>, 0.5 }
    cylinder {  m*<-3.9734847967676035,-7.364955980855061,-2.4828464446724636>, <0.5221807715530196,1.1625541602036464,0.17461898921340124>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    