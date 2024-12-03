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
    sphere { m*<-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 1 }        
    sphere {  m*<0.8999913096558815,0.3387343248347121,9.359302064164597>, 1 }
    sphere {  m*<8.267778507978683,0.05364207404245058,-5.211375364909334>, 1 }
    sphere {  m*<-6.628184685710312,6.576723447663092,-3.7205684617277273>, 1}
    sphere { m*<-3.489173895204627,-7.119279687598859,-1.8653569943378527>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8999913096558815,0.3387343248347121,9.359302064164597>, <-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 0.5 }
    cylinder { m*<8.267778507978683,0.05364207404245058,-5.211375364909334>, <-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 0.5}
    cylinder { m*<-6.628184685710312,6.576723447663092,-3.7205684617277273>, <-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 0.5 }
    cylinder {  m*<-3.489173895204627,-7.119279687598859,-1.8653569943378527>, <-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 0.5}

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
    sphere { m*<-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 1 }        
    sphere {  m*<0.8999913096558815,0.3387343248347121,9.359302064164597>, 1 }
    sphere {  m*<8.267778507978683,0.05364207404245058,-5.211375364909334>, 1 }
    sphere {  m*<-6.628184685710312,6.576723447663092,-3.7205684617277273>, 1}
    sphere { m*<-3.489173895204627,-7.119279687598859,-1.8653569943378527>, 1 }    

        pigment { color rgb<0.8,0,0>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }
    }    
    union {
    cylinder { m*<0.8999913096558815,0.3387343248347121,9.359302064164597>, <-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 0.5 }
    cylinder { m*<8.267778507978683,0.05364207404245058,-5.211375364909334>, <-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 0.5}
    cylinder { m*<-6.628184685710312,6.576723447663092,-3.7205684617277273>, <-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 0.5 }
    cylinder {  m*<-3.489173895204627,-7.119279687598859,-1.8653569943378527>, <-0.5191761845442804,-0.6512045890452053,-0.48998803287055104>, 0.5}

    pigment { color rgb<0.6,.2,.2>  }
    finish { ambient 0.1 diffuse 0.7 phong 1 }    
    }
    
    // rotate <0, 0, 0>
    }
    